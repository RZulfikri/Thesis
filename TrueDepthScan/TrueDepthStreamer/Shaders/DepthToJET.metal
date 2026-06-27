/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Metal compute shader that translates depth values to JET RGB values.
*/

#include <metal_stdlib>
using namespace metal;

struct BGRAPixel {
    uchar b;
    uchar g;
    uchar r;
    uchar a;
};

struct JETParams {
    int histogramSize;
    int binningFactor;
};

// Compute kernel
kernel void depthToJET(texture2d<float, access::read>  inputTexture  [[ texture(0) ]],
                       texture2d<float, access::write> outputTexture [[ texture(1) ]],
                       constant JETParams&  params     [[ buffer(0) ]],
                       constant float*      histogram  [[ buffer(1) ]],
                       constant BGRAPixel*  colorTable [[ buffer(2) ]],
                       uint2 gid [[ thread_position_in_grid ]])
{
    if ((gid.x >= inputTexture.get_width()) || (gid.y >= inputTexture.get_height())) {
        return;
    }

    float depth = inputTexture.read(gid).x;

    // Match export depth range exactly so preview and export show the same region
    constexpr float minDepth = 0.10;  // 10 cm
    constexpr float maxDepth = 0.50;  // 50 cm

    if (isnan(depth) || depth <= minDepth || depth >= maxDepth) {
        // Outside valid range → transparent, video shows through
        outputTexture.write(float4(0.0, 0.0, 0.0, 0.0), gid);
        return;
    }

    // Map depth to histogram bin using the same formula as HistogramCalculator
    // (binningFactor = 8000, so 0.10m → bin 800, 0.50m → bin 4000)
    int bin = clamp(int(depth * float(params.binningFactor)), 0, params.histogramSize - 1);

    // Look up histogram-equalized color index (HistogramCalculator outputs CDF → color index)
    int colorIdx = clamp(int(histogram[bin]), 0, 511);

    BGRAPixel pixel = colorTable[colorIdx];

    // Write as RGBA (note: BGRAPixel stores B,G,R,A but we output R,G,B,A to the texture)
    outputTexture.write(float4(float(pixel.r) / 255.0,
                               float(pixel.g) / 255.0,
                               float(pixel.b) / 255.0,
                               1.0), gid);
}
