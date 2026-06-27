/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
A view implementing point cloud rendering
*/

#import <MetalKit/MetalKit.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include "PointCloudMetalView.h"
#import "AAPLTransforms.h"
#include <simd/simd.h>
#import <Accelerate/Accelerate.h>

struct Uniforms {
    matrix_float4x4 viewMatrix;
    matrix_float3x3 cameraIntrinsics;
    vector_float2 imageDimensions;
};

simd::float3 matrix4_mul_vector3(simd::float4x4 m, simd::float3 v) {
    simd::float4 temp = { v.x, v.y, v.z, 0.0f };
    temp = simd_mul(m, temp);
    return { temp.x, temp.y, temp.z };
}

@implementation PointCloudMetalView {
    dispatch_queue_t _syncQueue;
    AVDepthData* _internalDepthFrame;
    CVPixelBufferRef _internalColorTexture;
    simd::float3 _center;   // current point camera looks at
    simd::float3 _eye;      // current camera position
    simd::float3 _up;       // camera "up" direction

    id<MTLCommandQueue> _commandQueue;
    id<MTLRenderPipelineState> _renderPipelineState;
    id<MTLDepthStencilState> _depthStencilState;
    
    // Textures
    CVMetalTextureCacheRef _textureCache;
    id<MTLTexture> _depthTexture;
    id<MTLTexture> _colorTexture;
    
    NSUInteger _vertexCount;
}

- (nonnull instancetype)initWithFrame:(CGRect)frameRect device:(nullable id<MTLDevice>)device {
    self = [super initWithFrame:frameRect device:device];
    [self internalInit];
    return self;
}

- (nonnull instancetype)initWithCoder:(nonnull NSCoder *)coder {
    self = [super initWithCoder:coder];
    self.device = MTLCreateSystemDefaultDevice();
    [self internalInit];
    return self;
}

- (void)internalInit {
    dispatch_queue_attr_t attr = NULL;
    attr = dispatch_queue_attr_make_with_autorelease_frequency(attr, DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM);
    attr = dispatch_queue_attr_make_with_qos_class(attr, QOS_CLASS_USER_INITIATED, 0);
    _syncQueue = dispatch_queue_create("PointCloudMetalView sync queue", attr);
    
    CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, self.device, nil, &_textureCache);
    
    [self configureMetal];
    
    self.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    self.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    
    [self resetView];
}

- (void)configureMetal {
    // Shader source embedded directly to avoid library loading issues
    NSString* shaderSource = @""
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "struct Uniforms {\n"
    "    float4x4 viewMatrix;\n"
    "    float3x3 cameraIntrinsics;\n"
    "    float2 imageDimensions;\n"
    "};\n"
    "\n"
    "struct VertexOut {\n"
    "    float4 position [[position]];\n"
    "    float4 color;\n"
    "    float pointSize [[point_size]];\n"
    "};\n"
    "\n"
    "vertex VertexOut vertexShaderPoints(uint vertexID [[ vertex_id ]],\n"
    "                                    texture2d<float, access::read> depthTexture [[ texture(0) ]],\n"
    "                                    texture2d<float, access::read> colorTexture [[ texture(1) ]],\n"
    "                                    constant Uniforms &uniforms [[ buffer(0) ]])\n"
    "{\n"
    "    VertexOut out;\n"
    "    \n"
    "    uint width = depthTexture.get_width();\n"
    "    uint height = depthTexture.get_height();\n"
    "    \n"
    "    // Grid position\n"
    "    uint x = vertexID % width;\n"
    "    uint y = vertexID / width;\n"
    "    \n"
    "    if (y >= height) {\n"
    "        out.position = float4(0, 0, 0, 1);\n"
    "        out.pointSize = 0;\n"
    "        return out;\n"
    "    }\n"
    "    \n"
    "    // Read depth\n"
    "    float depth = depthTexture.read(uint2(x, y)).r;\n"
    "    \n"
    "    // Filter invalid depth (0 or too far)\n"
    "    if (depth < 0.1 || depth > 2.0) {\n"
    "        out.position = float4(0, 0, 0, 1);\n"
    "        out.pointSize = 0;\n"
    "        return out;\n"
    "    }\n"
    "    \n"
    "    // Unproject to 3D\n"
    "    // x_camera = (x_pixel - cx) * depth / fx\n"
    "    // y_camera = (y_pixel - cy) * depth / fy\n"
    "    \n"
    "    float3x3 intrinsics = uniforms.cameraIntrinsics;\n"
    "    float fx = intrinsics[0][0];\n"
    "    float fy = intrinsics[1][1];\n"
    "    float cx = intrinsics[2][0];\n"
    "    float cy = intrinsics[2][1];\n"
    "    \n"
    "    float X = (float(x) - cx) * depth / fx;\n"
    "    float Y = (float(y) - cy) * depth / fy;\n"
    "    float Z = depth * 1000.0; // Scale to match view logic (mm)\n"
    "    X *= 1000.0;\n"
    "    Y *= 1000.0;\n"
    "    \n"
    "    float4 pos = float4(X, Y, Z, 1.0);\n"
    "    out.position = uniforms.viewMatrix * pos;\n"
    "    \n"
    "    // Sample color\n"
    "    // Map depth coordinates to color texture coordinates\n"
    "    // Assuming simple scaling for now\n"
    "    uint colorWidth = colorTexture.get_width();\n"
    "    uint colorHeight = colorTexture.get_height();\n"
    "    \n"
    "    uint cX = uint(float(x) / float(width) * float(colorWidth));\n"
    "    uint cY = uint(float(y) / float(height) * float(colorHeight));\n"
    "    \n"
    "    out.color = colorTexture.read(uint2(cX, cY));\n"
    "    out.pointSize = 8.0;\n"
    "    \n"
    "    return out;\n"
    "}\n"
    "\n"
    "fragment float4 fragmentShaderPoints(VertexOut in [[stage_in]])\n"
    "{\n"
    "    return in.color;\n"
    "}\n";
    
    NSError* error = nil;
    id<MTLLibrary> library = [self.device newLibraryWithSource:shaderSource options:nil error:&error];
    
    if (!library) {
        NSLog(@"Failed to compile shader: %@", error);
        return;
    }
    
    id <MTLFunction> vertexFunction = [library newFunctionWithName:@"vertexShaderPoints"];
    id <MTLFunction> fragmentFunction = [library newFunctionWithName:@"fragmentShaderPoints"];

    MTLRenderPipelineDescriptor *pipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineStateDescriptor.label = @"PointCloud Pipeline";
    pipelineStateDescriptor.vertexFunction = vertexFunction;
    pipelineStateDescriptor.fragmentFunction = fragmentFunction;
    pipelineStateDescriptor.colorAttachments[0].pixelFormat = self.colorPixelFormat;
    pipelineStateDescriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    
    MTLDepthStencilDescriptor *piplineDepthDescriptor = [[MTLDepthStencilDescriptor alloc] init];
    piplineDepthDescriptor.depthWriteEnabled = true;
    piplineDepthDescriptor.depthCompareFunction = MTLCompareFunctionLess;
    _depthStencilState = [self.device newDepthStencilStateWithDescriptor:piplineDepthDescriptor];
    
    _renderPipelineState = [self.device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor
                                                                       error:&error];

    if (!_renderPipelineState) {
        NSLog(@"Failed to created pipeline state, error %@", error);
    }
    
    _commandQueue = [self.device newCommandQueue];
}

- (void)setDepthFrame:(AVDepthData* _Nonnull)depth withTexture:(_Nonnull CVPixelBufferRef)unormTexture {
    dispatch_sync(_syncQueue, ^{
        self->_internalDepthFrame = depth;
        CVPixelBufferRelease(self->_internalColorTexture);
        self->_internalColorTexture = unormTexture;
        CVPixelBufferRetain(self->_internalColorTexture);
        
        [self updateTextures];
    });
    
    dispatch_async(dispatch_get_main_queue(), ^{
        [self setNeedsDisplay];
    });
}

- (void)updateTextures {
    if (!_internalDepthFrame || !_internalColorTexture) return;
    
    CVPixelBufferRef depthBuffer = _internalDepthFrame.depthDataMap;
    CVPixelBufferRef colorBuffer = _internalColorTexture;
    
    size_t width = CVPixelBufferGetWidth(depthBuffer);
    size_t height = CVPixelBufferGetHeight(depthBuffer);
    
    // Create Depth Texture
    // Fallback approach: Create MTLTexture manually and copy bytes
    // This avoids CVMetalTextureCache issues with certain pixel formats
    
    MTLTextureDescriptor *depthDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                         width:width
                                                                                        height:height
                                                                                     mipmapped:NO];
    depthDesc.usage = MTLTextureUsageShaderRead;
    _depthTexture = [self.device newTextureWithDescriptor:depthDesc];
    
    CVPixelBufferLockBaseAddress(depthBuffer, kCVPixelBufferLock_ReadOnly);
    void *depthBase = CVPixelBufferGetBaseAddress(depthBuffer);
    size_t depthBytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer);
    OSType depthFormat = CVPixelBufferGetPixelFormatType(depthBuffer);
    
    // We need to convert whatever we have to Float32 for the texture if it's not
    // Or we can use R16Float if source is 16-bit
    
    if (depthFormat == kCVPixelFormatType_DepthFloat32) {
        [_depthTexture replaceRegion:MTLRegionMake2D(0, 0, width, height)
                         mipmapLevel:0
                           withBytes:depthBase
                         bytesPerRow:depthBytesPerRow];
    } else {
        // Assume Float16 (native or 'hdep')
        // Convert to Float32 buffer for upload, or upload as R16Float
        // Let's recreate texture as R16Float if needed
        if (_depthTexture.pixelFormat != MTLPixelFormatR16Float) {
             MTLTextureDescriptor *desc16 = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                                                               width:width
                                                                                              height:height
                                                                                           mipmapped:NO];
             desc16.usage = MTLTextureUsageShaderRead;
             _depthTexture = [self.device newTextureWithDescriptor:desc16];
        }
        
        [_depthTexture replaceRegion:MTLRegionMake2D(0, 0, width, height)
                         mipmapLevel:0
                           withBytes:depthBase
                         bytesPerRow:depthBytesPerRow];
    }
    CVPixelBufferUnlockBaseAddress(depthBuffer, kCVPixelBufferLock_ReadOnly);
    
    // Create Color Texture
    // Using TextureCache for color usually works fine, but let's be consistent and robust
    size_t colorWidth = CVPixelBufferGetWidth(colorBuffer);
    size_t colorHeight = CVPixelBufferGetHeight(colorBuffer);
    
    MTLTextureDescriptor *colorDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                         width:colorWidth
                                                                                        height:colorHeight
                                                                                     mipmapped:NO];
    colorDesc.usage = MTLTextureUsageShaderRead;
    _colorTexture = [self.device newTextureWithDescriptor:colorDesc];
    
    CVPixelBufferLockBaseAddress(colorBuffer, kCVPixelBufferLock_ReadOnly);
    [_colorTexture replaceRegion:MTLRegionMake2D(0, 0, colorWidth, colorHeight)
                     mipmapLevel:0
                       withBytes:CVPixelBufferGetBaseAddress(colorBuffer)
                     bytesPerRow:CVPixelBufferGetBytesPerRow(colorBuffer)];
    CVPixelBufferUnlockBaseAddress(colorBuffer, kCVPixelBufferLock_ReadOnly);
    
    _vertexCount = width * height;
}

- (void)drawRect:(CGRect)rect {
    if (_vertexCount == 0 || !_depthTexture || !_colorTexture) return;
    
    id <MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    MTLRenderPassDescriptor *renderPassDescriptor = self.currentRenderPassDescriptor;
    
    if(renderPassDescriptor != nil) {
        renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
        
        renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionClear;
        renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionStore;
        renderPassDescriptor.depthAttachment.clearDepth = 1.0;
        
        id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        [renderEncoder setDepthStencilState:_depthStencilState];
        [renderEncoder setRenderPipelineState:_renderPipelineState];

        // Prepare Uniforms
        struct Uniforms uniforms;
        uniforms.viewMatrix = [self getFinalViewMatrix];
        uniforms.cameraIntrinsics = _internalDepthFrame.cameraCalibrationData.intrinsicMatrix;
        
        // Handle intrinsics scaling if needed
        size_t depthWidth = _depthTexture.width;
        CGSize refDims = _internalDepthFrame.cameraCalibrationData.intrinsicMatrixReferenceDimensions;
        if (refDims.width > 0 && refDims.width != depthWidth) {
            float ratio = (float)depthWidth / refDims.width;
            uniforms.cameraIntrinsics.columns[0][0] *= ratio; // fx
            uniforms.cameraIntrinsics.columns[1][1] *= ratio; // fy
            uniforms.cameraIntrinsics.columns[2][0] *= ratio; // ox
            uniforms.cameraIntrinsics.columns[2][1] *= ratio; // oy
        }
        
        uniforms.imageDimensions = (vector_float2){(float)depthWidth, (float)_depthTexture.height};

        [renderEncoder setVertexBytes:&uniforms length:sizeof(uniforms) atIndex:0];
        [renderEncoder setFragmentTexture:_depthTexture atIndex:0]; // Actually vertex shader reads it
        [renderEncoder setVertexTexture:_depthTexture atIndex:0];
        [renderEncoder setVertexTexture:_colorTexture atIndex:1];

        [renderEncoder drawPrimitives:MTLPrimitiveTypePoint
                          vertexStart:0
                          vertexCount:_vertexCount];
        
        [renderEncoder endEncoding];
        [commandBuffer presentDrawable:self.currentDrawable];
    }
    
    [commandBuffer commit];
}

- (void)rollAroundCenter:(float)angle {
    dispatch_sync(_syncQueue, ^{
        simd::float3 viewDir = self->_center - self->_eye;
        viewDir = simd::normalize(viewDir);
        simd::float4x4 rotMat = AAPL::rotate(angle, viewDir);
        self->_up = matrix4_mul_vector3(rotMat, self->_up);
    });
}

// rotate around Y axis
- (void)yawAroundCenter:(float)angle {
    dispatch_sync(_syncQueue, ^{
        simd::float4x4 rotMat = AAPL::rotate(angle, self->_up);

        self->_eye = self->_eye - self->_center;
        self->_eye = matrix4_mul_vector3(rotMat, self->_eye);
        self->_eye = self->_eye + self->_center;

        self->_up = matrix4_mul_vector3(rotMat, self->_up);
    });
}

// rotate around X axis
- (void)pitchAroundCenter:(float)angle {
    dispatch_sync(_syncQueue, ^{
        simd::float3 viewDirection = simd_normalize(self->_center - self->_eye);
        simd::float3 rightVector = simd_cross(self->_up, viewDirection);
        
        simd::float4x4 rotMat = AAPL::rotate(angle, rightVector);

        self->_eye = self->_eye - self->_center;
        self->_eye = matrix4_mul_vector3(rotMat, self->_eye);
        self->_eye = self->_eye + self->_center;

        self->_up = matrix4_mul_vector3(rotMat, self->_up);
    });
}

- (void)moveTowardCenter:(float)scale
{
    __block float _scale = scale;
    
    dispatch_sync(_syncQueue, ^{
        simd::float3 direction = self->_center - self->_eye;
        
        // don't move to the other side of _center
        float distance = sqrt(simd_dot(direction, direction));
        if (_scale > distance)
            _scale = distance - 3.0;
        
        direction = simd::normalize(direction);
        direction = direction * _scale;
        self->_eye += direction;
    });
}

-(void)resetView {
    dispatch_sync(_syncQueue, ^{
        self->_center = simd::float3 { 0, 0, 500 };   // start at a distance of ~50cm
        self->_eye = simd::float3 { 0, 0, 0 };
        // The TrueDepth camera outputs frames that are aligned to device landscape, so should be rotated 90 degrees counter-clockwise
        self->_up = simd_float3 { -1 , 0 ,0 };
        
    });
}

-(simd::float4x4)getFinalViewMatrix {
    float aspect = (self.drawableSize.width / self.drawableSize.height);

    // Use a magic number that simply looks good
    float vfov = 70;
    simd::float4x4 appleProjMat = AAPL::perspective_fov(vfov, aspect, 0.01f, 30000);

    __block simd::float3 eye, center, up;
    
    // take camera position in a synchornized way
    dispatch_sync(_syncQueue, ^{
        eye = self->_eye;
        center = self->_center;
        up = self->_up;
    });
                  
    simd::float4x4 appleViewMat = AAPL::lookAt(eye, center, up);
    
    // Final view matrix is projection * view * model. In our case, we never move the model itself, so we can ignore its matrix.
    return appleProjMat * appleViewMat;
}

@end
