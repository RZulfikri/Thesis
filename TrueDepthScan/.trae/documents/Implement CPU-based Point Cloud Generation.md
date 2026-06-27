# Point Cloud Rendering Plan (CPU-Based)

Since the direct `CVMetalTextureCacheCreateTextureFromImage` approach is failing for the specific depth pixel format (`hdep` / 1751410032), I will switch to a **CPU-based buffer generation strategy**. Instead of trying to create a Metal texture directly from the `AVDepthData` CVPixelBuffer, I will manually extract the depth points on the CPU, convert them into a vertex buffer, and then render that buffer using Metal.

This approach aligns with the user's request to "extract depth data," "convert to pointcloud," and "display."

## 1. Filter Depth Data (CPU)
- [ ] **Modify `PointCloudMetalView.mm`**:
    - Remove the failing `CVMetalTextureCacheCreateTextureFromImage` logic for the depth texture.
    - Implement a method to iterate over the depth buffer (on CPU).
    - **Filter**: Only select points with valid depth (non-zero/non-NaN). The user mentioned "highest heatmap object" — strictly speaking, this usually means "closest objects" (low depth values) in a depth map context. I will implement a threshold to ignore background or invalid points (e.g., depth > 0 and depth < max_dist).
    - **Unproject**: Convert the (u, v, z) coordinates to (x, y, z) 3D points using the camera intrinsics *on the CPU*.
    - **Color**: Sample the corresponding color from the video texture for each point.

## 2. Generate Vertex Buffer
- [ ] **Create Vertex Structure**: Define a simple struct `ParticleVertex { vector_float3 position; vector_float4 color; }`.
- [ ] **Populate Buffer**:
    - Create an `MTLBuffer` to hold these vertices.
    - Fill it with the unprojected points and their colors.
    - This replaces the need for a depth texture in the vertex shader.

## 3. Update Rendering Pipeline
- [ ] **Update Shaders (`Shaders.metal`)**:
    - Modify the vertex shader to accept a buffer of `ParticleVertex` instead of a 2D depth texture.
    - It should pass the position and color directly to the fragment shader.
- [ ] **Update `PointCloudMetalView.mm`**:
    - In `drawRect`, instead of binding textures, bind the new `MTLBuffer` containing the point cloud.
    - Call `drawPrimitives` with the count of valid points found.

## 4. Verification
- [ ] **Compile & Run**: Ensure no Metal validation errors.
- [ ] **Test**: Verify that the captured point cloud is visible and looks correct (matches the 3D structure of the scene).

This method bypasses the texture creation issue entirely and gives us full control over which points to render (filtering).
