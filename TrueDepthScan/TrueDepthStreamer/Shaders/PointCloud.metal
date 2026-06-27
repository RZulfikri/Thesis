/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Metal shaders used for point-cloud view
*/

#include <metal_stdlib>
using namespace metal;

struct ParticleVertex {
    float3 position;
    float4 color;
};

typedef struct
{
    float4 clipSpacePosition [[position]];
    float pSize [[point_size]];
    float4 color;
} RasterizerDataColor;

// Vertex Function
vertex RasterizerDataColor
vertexShaderPoints(uint vertexID [[ vertex_id ]],
                   constant ParticleVertex *vertices [[ buffer(0) ]],
                   constant float4x4& viewMatrix [[ buffer(1) ]])
{
    RasterizerDataColor out;
    
    ParticleVertex v = vertices[vertexID];
    
    float4 xyzw = { v.position.x, v.position.y, v.position.z, 1.f };
    
    out.clipSpacePosition = viewMatrix * xyzw;
    out.color = v.color;
    out.pSize = 10.0f; // Increased point size for visibility
    
    return out;
}

fragment float4 fragmentShaderPoints(RasterizerDataColor in [[stage_in]])
{
    return in.color;
}
