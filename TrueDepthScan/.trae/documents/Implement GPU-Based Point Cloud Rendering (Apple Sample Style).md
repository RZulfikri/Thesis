# GPU-Based Point Cloud Preview Plan

I will implement the point cloud rendering technique described in the Apple Developer documentation (Scene Depth Point Cloud), adapting it for `AVDepthData`. This shifts the processing from CPU to GPU for better performance and standardizes the implementation.

## 1. Texture Management (`PointCloudMetalView.mm`)
- [ ] **Texture Creation Strategy**:
    - Instead of creating a buffer of vertices on the CPU, I will create two `MTLTexture` objects: one for **Depth** and one for **Color**.
    - **Robustness**: I will implement a fallback mechanism. I'll first attempt to create textures efficiently using `CVMetalTextureCache`. If that fails (as seen previously with the specific DepthFloat16 format), I will manually create a standard `MTLTexture` and upload the pixel buffer bytes using `replaceRegion`. This ensures compatibility with any pixel format.
- [ ] **Render Pipeline Update**:
    - Update `drawRect` to bind these two textures to the fragment/vertex shaders.
    - Pass the **Camera Intrinsics** (fx, fy, ox, oy) and **Image Dimensions** as a uniform buffer to the shader, allowing the GPU to perform the unprojection (2D -> 3D).

## 2. Shader Implementation (`PointCloud.metal`)
- [ ] **Rewrite Shaders**:
    - **Vertex Shader**:
        - Accept `texture2d<float> depthMap` and `texture2d<float> colorMap`.
        - Accept `uniforms` (intrinsics, dimensions).
        - Use `vertex_id` to calculate the (x, y) pixel coordinates.
        - Sample the depth value.
        - Unproject to 3D space: `z = depth`, `x = (u - ox) * z / fx`, `y = (v - oy) * z / fy`.
        - Pass the 3D position and sampled color to the rasterizer.
    - **Fragment Shader**:
        - Simply output the color passed from the vertex shader.
- [ ] **Grid Generation**:
    - The draw call will request `width * height` points. The vertex shader maps each index to a pixel.

## 3. Verification
- [ ] **Compile**: Since I'm embedding shaders dynamically (due to the library loading issue), I will update the embedded shader source string.
- [ ] **Run**: Verify that the point cloud appears and matches the captured object.

This approach aligns exactly with the "Displaying a point cloud using scene depth" methodology, using the GPU for geometry generation.
