import Foundation
import os.log

private let apiLog = OSLog(subsystem: Bundle.main.bundleIdentifier ?? "TrueDepthStreamer", category: "ApiService")

final class ApiService {
    static let shared = ApiService()
    private init() {}

    private let maxUploadDepthFiles = 8
    
    enum ApiError: Error {
        case invalidURL
        case httpStatus(Int, String)
        case noResponse
    }
    
    private var overrideBaseURLString: String?
    private var overrideAPIKey: String?
    
    func setBaseURL(_ url: String?) {
        overrideBaseURLString = url?.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    func setAPIKey(_ key: String?) {
        overrideAPIKey = key?.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    private var baseURLString: String {
        if let o = overrideBaseURLString, !o.isEmpty {
            return o.hasSuffix("/") ? String(o.dropLast()) : o
        }
        let plistBase = (Bundle.main.object(forInfoDictionaryKey: "BASE_URL") as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let envBase = ProcessInfo.processInfo.environment["FUNCTION_BASE_URL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let fallback = "https://us-central1-palmrecognition-d10ca.cloudfunctions.net"
        let raw = (plistBase?.isEmpty == false ? plistBase : envBase) ?? fallback
        return raw.hasSuffix("/") ? String(raw.dropLast()) : raw
    }
    
    private var apiKey: String? {
        if let k = overrideAPIKey, !k.isEmpty { return k }
        let plistKey = (Bundle.main.object(forInfoDictionaryKey: "APP_API_KEY") as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let envKey = ProcessInfo.processInfo.environment["APP_API_KEY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let key = (plistKey?.isEmpty == false ? plistKey : envKey) ?? ""
        return key.isEmpty ? nil : key
    }
    
    private func endpoint(for path: String) -> URL? {
        let p = path.hasPrefix("/") ? String(path.dropFirst()) : path
        return URL(string: "\(baseURLString)/\(p)")
    }
    
    func uploadRegistrationOpen3D(folderURL: URL, uuid: String, label: String, completion: @escaping (Result<Void, Error>) -> Void) {
        guard let url = endpoint(for: "run_registration") else {
            completion(.failure(ApiError.invalidURL))
            return
        }
        
        var files: [(fieldName: String, filename: String, mime: String, data: Data)] = []
        let fileManager = FileManager.default
        do {
            let contents = try fileManager.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
            let allNames = contents.map { $0.lastPathComponent }.sorted()
            os_log(.info, log: apiLog, "Open3D upload folder: %{public}@ contents: %{public}@",
                   folderURL.path, allNames.joined(separator: ", "))

            let calibURL = folderURL.appendingPathComponent("calibration.json")
            if let d = try? Data(contentsOf: calibURL) {
                files.append(("files", "calibration.json", "application/json", d))
                os_log(.info, log: apiLog, "Attached calibration.json (%d bytes)", d.count)
            } else {
                os_log(.error, log: apiLog, "Missing calibration.json at %{public}@", calibURL.path)
            }

            let depthFiles = contents
                .filter { $0.lastPathComponent.hasPrefix("depth") && $0.pathExtension == "bin" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }

            let uploadCount = min(maxUploadDepthFiles, depthFiles.count)
            os_log(.info, log: apiLog, "Found %d depth bins, uploading %d",
                   depthFiles.count, uploadCount)

            for i in 0..<uploadCount {
                let dpth = depthFiles[i]
                if let dd = try? Data(contentsOf: dpth) {
                    files.append(("files", dpth.lastPathComponent, "application/octet-stream", dd))
                    os_log(.info, log: apiLog, "Attached depth: %{public}@ (%d bytes)",
                           dpth.lastPathComponent, dd.count)
                } else {
                    os_log(.error, log: apiLog, "Failed reading depth bin at index %d", i)
                }
            }
        } catch {
            os_log(.error, log: apiLog, "Failed to list Open3D folder %{public}@: %{public}@",
                   folderURL.path, error.localizedDescription)
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 540.0
        if let k = apiKey, !k.isEmpty {
            request.setValue(k, forHTTPHeaderField: "X-Api-Key")
        }
        
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        
        func appendFormField(name: String, value: String) {
            guard
                let boundaryData = "--\(boundary)\r\n".data(using: .utf8),
                let dispositionData = "Content-Disposition: form-data; name=\"\(name)\"\r\n".data(using: .utf8),
                let lineBreak = "\r\n".data(using: .utf8),
                let valueData = "\(value)\r\n".data(using: .utf8)
            else { return }
            
            body.append(boundaryData)
            body.append(dispositionData)
            body.append(lineBreak)
            body.append(valueData)
        }
        
        func appendFile(field: String, filename: String, mime: String, data: Data) {
            guard
                let boundaryData = "--\(boundary)\r\n".data(using: .utf8),
                let dispositionData = "Content-Disposition: form-data; name=\"\(field)\"; filename=\"\(filename)\"\r\n".data(using: .utf8),
                let typeData = "Content-Type: \(mime)\r\n".data(using: .utf8),
                let lineBreak = "\r\n".data(using: .utf8)
            else { return }
            
            body.append(boundaryData)
            body.append(dispositionData)
            body.append(typeData)
            body.append(lineBreak)
            body.append(data)
            body.append(lineBreak)
        }
        
        appendFormField(name: "uuid", value: uuid)
        appendFormField(name: "label", value: label)
        for f in files {
            appendFile(field: f.fieldName, filename: f.filename, mime: f.mime, data: f.data)
        }
        if let end = "--\(boundary)--\r\n".data(using: .utf8) {
            body.append(end)
        }
        request.httpBody = body
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                os_log(.error, log: apiLog, "API error: %{public}@", error.localizedDescription)
                completion(.failure(error))
                return
            }
            guard let http = response as? HTTPURLResponse else {
                os_log(.error, log: apiLog, "API error: No HTTP response")
                completion(.failure(ApiError.noResponse))
                return
            }
            var bodyText: String?
            if let data = data {
                if let obj = try? JSONSerialization.jsonObject(with: data),
                   let pretty = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted]),
                   let s = String(data: pretty, encoding: .utf8) {
                    bodyText = s
                } else {
                    bodyText = String(data: data, encoding: .utf8)
                }
            }
            os_log(.info, log: apiLog, "API status: %d", http.statusCode)
            if let t = bodyText {
                os_log(.info, log: apiLog, "API response: %{public}@", t)
            }
            if http.statusCode != 200 {
                let text = data.flatMap { String(data: $0, encoding: .utf8) } ?? "Unknown error"
                completion(.failure(ApiError.httpStatus(http.statusCode, text)))
                return
            }
            completion(.success(()))
        }
        task.resume()
    }
}
