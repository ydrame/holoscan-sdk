#version 450 core

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_normal;


vec4 lightSource = vec4(2.0, 2.0, 20.0, 0.0);

layout(location = 0) out vec4 vVaryingColor;

layout(push_constant) uniform constants
{
    mat4x4 modelviewMatrix;
    mat4x4 modelviewprojectionMatrix;
    mat3x3 normalMatrix;
} pushConstants;

void main()
{
    gl_Position = pushConstants.modelviewprojectionMatrix * in_position;
    vec3 vEyeNormal = pushConstants.normalMatrix * in_normal;
    vec4 vPosition4 = pushConstants.modelviewMatrix * in_position;
    vec3 vPosition3 = vPosition4.xyz / vPosition4.w;
    vec3 vLightDir = normalize(lightSource.xyz - vPosition3);
    float diff = max(0.0, dot(vEyeNormal, vLightDir));
    vVaryingColor = vec4(diff * in_color.rgb, 1.0);
}
