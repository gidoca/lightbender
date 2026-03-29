#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in mat3 fragTBN;

layout(location = 0) out vec4 outColor;

struct GpuLight {
    vec4  positionOrDirection;
    vec3  color;
    float intensity;
    float range;
    float _pad0;
    vec2  spotAngles;
    vec4  _pad1;
};

layout(set = 0, binding = 0) uniform FrameUniforms {
    mat4     view;
    mat4     projection;
    vec4     cameraPosition;
    GpuLight lights[8];
    uint     lightCount;
    float    envIntensity;
} frame;

// Material textures (set 1)
layout(set = 1, binding = 0) uniform sampler2D texBaseColor;
layout(set = 1, binding = 1) uniform sampler2D texNormal;
layout(set = 1, binding = 2) uniform sampler2D texMetallicRoughness;
layout(set = 1, binding = 3) uniform sampler2D texOcclusion;
layout(set = 1, binding = 4) uniform sampler2D texEmissive;

// Environment map (set 2)
layout(set = 2, binding = 0) uniform sampler2D envMap;

const float PI = 3.14159265359;

// ── PBR helper functions ──────────────────────────────────────────────────────

float distributionGGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(float NdotV, float NdotL, float roughness) {
    return geometrySchlickGGX(NdotV, roughness)
         * geometrySchlickGGX(NdotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec2 directionToEquirect(vec3 dir) {
    float phi   = atan(dir.z, dir.x);
    float theta = asin(clamp(dir.y, -1.0, 1.0));
    return vec2(phi / (2.0 * PI) + 0.5, -theta / PI + 0.5);
}

// ── Main ──────────────────────────────────────────────────────────────────────

void main() {
    // Sample textures
    vec4 baseColorSample    = texture(texBaseColor, fragUV);
    vec3 normalSample       = texture(texNormal, fragUV).rgb;
    vec2 metallicRoughness  = texture(texMetallicRoughness, fragUV).bg; // B=metallic, G=roughness
    float occlusion         = texture(texOcclusion, fragUV).r;
    vec3 emissive           = texture(texEmissive, fragUV).rgb;

    // Alpha cutout
    if (baseColorSample.a < 0.01) discard;

    vec3 albedo    = pow(baseColorSample.rgb, vec3(2.2)); // sRGB → linear
    float metallic  = metallicRoughness.x;
    float roughness = max(metallicRoughness.y, 0.04);

    // Normal mapping
    vec3 N = normalSample * 2.0 - 1.0;
    N = normalize(fragTBN * N);

    vec3 V = normalize(frame.cameraPosition.xyz - fragWorldPos);

    // Reflectance at normal incidence (F0)
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Accumulate radiance from all scene lights
    float NdotV = max(dot(N, V), 0.001);
    vec3 Lo = vec3(0.0);

    // Fallback: if no lights are defined, use a default sun
    uint numLights = frame.lightCount;
    if (numLights == 0u) {
        // Default directional light (warm sunlight)
        vec3 L = normalize(vec3(1.0, 2.0, 1.5));
        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);
        float D = distributionGGX(NdotH, roughness);
        float G = geometrySmith(NdotV, NdotL, roughness);
        vec3  F = fresnelSchlick(HdotV, F0);
        vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        Lo += (kD * albedo / PI + specular) * vec3(3.0, 2.85, 2.55) * NdotL;
    }

    for (uint i = 0u; i < numLights; i++) {
        GpuLight light = frame.lights[i];
        float lightType = light.positionOrDirection.w;

        vec3 L;
        float attenuation = 1.0;

        if (lightType < 0.5) {
            // Directional light (w=0): direction points toward the light source
            L = normalize(-light.positionOrDirection.xyz);
        } else {
            // Point (w=1) or spot (w=2) light
            vec3 toLight = light.positionOrDirection.xyz - fragWorldPos;
            float dist = length(toLight);
            L = toLight / max(dist, 0.0001);

            // Distance attenuation (inverse-square with range cutoff)
            float rangeAtt = max(1.0 - pow(dist / light.range, 4.0), 0.0);
            attenuation = (rangeAtt * rangeAtt) / max(dist * dist, 0.0001);

            // Spot cone attenuation (w=2)
            if (lightType > 1.5) {
                // spotAngles.x = cos(inner), spotAngles.y = cos(outer)
                // Note: direction for spot needs to be stored; for now use -L as approximation
                // TODO: store spot direction separately when GpuLight is extended
                float theta = dot(-L, normalize(-light.positionOrDirection.xyz));
                float epsilon = light.spotAngles.x - light.spotAngles.y;
                float spotAtt = clamp((theta - light.spotAngles.y) / max(epsilon, 0.0001), 0.0, 1.0);
                attenuation *= spotAtt;
            }
        }

        vec3 radiance = light.color * light.intensity * attenuation;

        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);

        float D = distributionGGX(NdotH, roughness);
        float G = geometrySmith(NdotV, NdotL, roughness);
        vec3  F = fresnelSchlick(HdotV, F0);

        vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }

    // Image-based lighting (IBL) from environment map
    vec3 irradiance = texture(envMap, directionToEquirect(N)).rgb;
    vec3 diffuseIBL = irradiance * albedo * (1.0 - metallic);

    vec3 R = reflect(-V, N);
    vec3 specEnv = texture(envMap, directionToEquirect(R)).rgb;
    vec3 F_ibl = fresnelSchlick(max(dot(N, V), 0.0), F0);
    vec3 specularIBL = specEnv * F_ibl;

    vec3 ambient = (diffuseIBL + specularIBL) * occlusion * frame.envIntensity;

    vec3 color = ambient + Lo + emissive * 3.0;

    // Reinhard tone mapping + gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, baseColorSample.a);
}
