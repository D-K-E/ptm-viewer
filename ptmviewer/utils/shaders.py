# author: Kaan Eraslan
# license: see, LICENSE.
# Simple file for holding shaders

lambertVshader = """
#version 330 core

in vec3 aPos;
in vec3 aNormal;
in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
}
"""

lambertFshader = """
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D diffuseMap;
uniform vec3 lightColor;
uniform vec3 lightPos;

void main() {
    vec3 surfaceNormal = normalize(Normal);
    vec3 lightDirection = normalize(lightPos - FragPos);
    vec3 diffuseColor = texture(diffuseMap, TexCoord).rgb;
    float costheta = max(dot(lightDirection, surfaceNormal), 0.0);

    // lambertian term: k_d * (N \cdot L) * I_p
    FragColor = vec4(diffuseColor * lightColor * costheta, 1.0);
}
"""

phongVshader = """
#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;

void main() 
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = aNormal;
}
"""

phongFshader = """
#version 330 core

in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform bool blinn;

uniform vec3 lightPos;
uniform float lightIntensity;
uniform vec3 lightColor;

uniform vec3 viewerPosition;

uniform vec3 ambientColor;
uniform float ambientCoeff;

uniform float specularCoeff;
uniform vec3 specularColor;

uniform float diffuseCoeff;
uniform vec3 diffuseColor;

uniform float shininess;

uniform float attC1;
uniform float attC2;
uniform float attC3;

void main() {
    // ambient term I_a × k_a × O_d
    vec3 ambientTerm = ambientColor * ambientCoeff * diffuseColor;

    // lambertian terms k_d * (N \cdot L) * I_p
    vec3 surfaceNormal = normalize(Normal);
    vec3 lightDirection = normalize(lightPos - FragPos);
    float costheta = dot(lightDirection, surfaceNormal);
    vec3 lambertianTerm = costheta * diffuseCoeff * lightIntensity * lightColor;

    // attenuation term f_att
    // f_att = min(\frac{1}{c_1 + c_2{\times}d_L + c_3{\times}d^2_{L}} , 1)
    float dist = distance(lightPos, FragPos);
    float distSqr = dist * dist;
    float att1 = dist * attC2;
    float att2 = distSqr * attC3;
    float result = attC1 + att2 + att1;
    result = 1 / result;
    float attenuation = min(result, 1);

    // expanding lambertian to phong
    vec3 phongTerms = lambertianTerm * attenuation;
    vec3 phong1 = phongTerms * diffuseColor;

    // phong adding specular terms
    vec3 phong2 = specularColor * specularCoeff;

    vec3 viewerDirection = normalize(viewerPosition - FragPos);
    float specAngle = 0.0;
    if(blinn) {
        vec3 halfwayDirection = normalize(lightDirection + viewerDirection);
        specAngle = max(dot(surfaceNormal, halfwayDirection), 0);
    }else{
        vec3 reflectionDirection = reflect(-lightDirection, surfaceNormal);
        specAngle = max(dot(viewerDirection, reflectionDirection), 0);
    }
    float specular = pow(specAngle, shininess);
    phong2 = phong2 * specular;
    FragColor = vec4(phong1 + phong2 + ambientTerm, 1.0);
}
"""

quadVshader = """
#version 330 core

// specify input
in vec3 aPos;
in vec3 aNormal;
in vec2 aTexCoord;
in vec3 aTangent;
in vec3 aBiTangent;

// specify output
out vec3 FragPos;
out vec2 TexCoords;
out vec3 TangentLightPos;
out vec3 TangentViewPos;
out vec3 TangentFragPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform vec3 lightPos;
uniform vec3 viewPos;

// function

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    TexCoords = aTexCoord;

    // create tangent bitangent normal matrix for
    // transforming light coordinates
    mat3 normMat = transpose(inverse(mat3(model)));
    vec3 TanV = normalize(normMat * aTangent);
    vec3 NormV = normalize(normMat * aNormal);
    TanV = normalize(TanV - dot(TanV, NormV) * NormV);
    vec3 BiTanV = cross(TanV, NormV);

    // computing light position and view position in tangent space
    mat3 TBN = transpose(mat3(TanV, BiTanV, NormV));
    TangentLightPos = TBN * lightPos;
    TangentViewPos = TBN * viewPos;
    TangentFragPos = TBN * FragPos;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

quadFshader = """
#version 330 core
in vec3 FragPos;
in vec2 TexCoords;
in vec3 TangentLightPos;
in vec3 TangentViewPos;
in vec3 TangentFragPos;

out vec4 FragColor;

uniform sampler2D diffuseMap; // object colors per fragment
uniform sampler2D normalMap; // normals per vertex

uniform float ambientCoeff;
uniform float shininess;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 viewPos;

void main()
{
    // obtain normal map from texture [0,1]
    vec3 normal = texture(normalMap, TexCoords).rgb;

    // transform vector to range [-1,1]
    normal = normalize((normal * 2.0) - 1.0);

    // get diffuse color
    vec3 color = texture(diffuseMap, TexCoords).rgb;

    // ambient color for object
    vec3 ambientColor = color * ambientCoeff;

    // costheta
    vec3 lightDir = normalize(TangentLightPos - TangentFragPos);
    float costheta = dot(lightDir, normal);
    costheta = max(costheta, 0.0);
    vec3 diffuseColor = color * costheta;

    // specular
    vec3 viewDir = normalize(TangentViewPos - TangentFragPos);
    // vec3 reflectDir = reflect(-lightDir, normal);
    vec3 halfway = normalize(lightDir + viewDir);
    float specAngle = dot(normal, halfway);
    specAngle = max(specAngle, 0.0);
    specAngle = pow(specAngle, shininess);
    vec3 specularColor = lightColor * specAngle;

    // final fragment color
    FragColor = vec4(ambientColor + diffuseColor + specularColor, 1.0);
}
"""

quadFshaderPerChannel = """
#version 330 core
in vec3 FragPos;
in vec2 TexCoords;
in vec3 TangentLightPos;
in vec3 TangentViewPos;
in vec3 TangentFragPos;

out vec4 FragColor;

uniform sampler2D diffuseMap; // object colors per fragment
uniform sampler2D normalMap1; // normals per vertex
uniform sampler2D normalMap2; // normals per vertex
uniform sampler2D normalMap3; // normals per vertex

uniform float ambientCoeff;
uniform float shininess;
uniform float attc1;
uniform float attc2;
uniform float attc3;
uniform float cutOff;

uniform vec3 lightPos;
uniform vec3 lightDirection;
uniform vec3 lightColor;
uniform vec3 viewPos;

float computeDiffColorPerChannel(vec3 normal, vec3 lightDir, float intensity);
float computeSpecColorPerChannel(vec3 normal, vec3 dirVec, 
                                 float shininess, float intensity);
vec3 computeSpecColor(vec3 normal1, vec3 normal2, vec3 normal3, vec3 dirVec,
                      float shininess, vec3 color);
vec3 computeDiffColor(vec3 normal1, vec3 normal2, vec3 normal3, vec3 lightDir,
                      vec3 color);
float computeAttenuation(float attC1, float attC2, float attC3, float dist);


void main()
{
    // obtain normal map from texture [0,1]
    vec3 normalr = texture(normalMap1, TexCoords).rgb;
    vec3 normalg = texture(normalMap2, TexCoords).rgb;
    vec3 normalb = texture(normalMap3, TexCoords).rgb;
    
    // get diffuse color for object
    vec3 color = texture(diffuseMap, TexCoords).rgb;

    // attenuation
    float distanceLightFrag = length(lightPos - FragPos);
    float att = computeAttenuation(attc1, attc2, attc3, distanceLightFrag);

    // ambient color for object
    vec3 ambientColor = color * ambientCoeff * att;


    // simple light direction
    vec3 lightDir = normalize(TangentLightPos - TangentFragPos);

    // spotlight cone theta
    float theta = dot(lightDir, lightDirection);
    if (theta > cutOff)
    {
        // costheta
        vec3 diffuseColor = computeDiffColor(normalr, normalg, normalb, lightDir,
                                             color) * att;
        // specular
        vec3 viewDir = normalize(TangentViewPos - TangentFragPos);
        // vec3 reflectDir = reflect(-lightDir, normal);
        vec3 halfway = normalize(lightDir + viewDir);
        vec3 specularColor = computeSpecColor(normalr, normalg, normalb, halfway,
                                              shininess, lightColor) * att;
        // final fragment color
        FragColor = vec4(ambientColor + diffuseColor + specularColor, 1.0);
    }else{
    FragColor = vec4(ambientColor, 1.0);
    }
}

float computeAttenuation(float attC1, float attC2, float attC3, float dist)
{
    /* f_att = min(\frac{1}{c_1 + c_2{\times}d_L + c_3{\times}d^2_{L}} , 1)
    */
    float distSqr = dist * dist;
    float att1 = dist * attC2;
    float att2 = distSqr * attC3;
    float result = attC1 + att2 + att1;
    result = 1 / result;
    return min(result, 1);
}

float computeDiffColorPerChannel(vec3 normal, vec3 lightDir, float intensity)
{
    // transform vector to range [-1,1]
    normal = normalize((normal * 2.0) - 1.0);
    float costheta = dot(lightDir, normal);
    costheta = max(costheta, 0.0);
    return costheta * intensity;
}
vec3 computeDiffColor(vec3 normal1, vec3 normal2, vec3 normal3, vec3 lightDir,
                      vec3 color)
{
    vec3 diffuseColor = vec3(1.0);
    diffuseColor.x = computeDiffColorPerChannel(normal1, lightDir, color.x);
    diffuseColor.y = computeDiffColorPerChannel(normal2, lightDir, color.y);
    diffuseColor.z = computeDiffColorPerChannel(normal3, lightDir, color.z);
    return diffuseColor;
}

float computeSpecColorPerChannel(vec3 normal, vec3 dirVec, 
                                 float shininess, float intensity)
{
    normal = normalize((normal * 2.0) - 1.0);
    float specAngle = dot(normal, dirVec);
    specAngle = max(specAngle, 0.0);
    specAngle = pow(specAngle, shininess);
    return specAngle * intensity;
}
vec3 computeSpecColor(vec3 normal1, vec3 normal2, vec3 normal3, vec3 dirVec,
                      float shininess, vec3 color)
{
    vec3 specularColor = vec3(1.0);
    specularColor.x = computeSpecColorPerChannel(normal1, dirVec,
                                                 shininess, lightColor.x);
    specularColor.y = computeSpecColorPerChannel(normal2, dirVec,
                                                 shininess, lightColor.y);
    specularColor.z = computeSpecColorPerChannel(normal3, dirVec,
                                                 shininess, lightColor.z);
    return specularColor;
}
"""
quadFshaderPerChannelTest = """
#version 330 core
in vec3 FragPos;
in vec2 TexCoords;
in vec3 TangentLightPos;
in vec3 TangentViewPos;
in vec3 TangentFragPos;

out vec4 FragColor;

uniform sampler2D diffuseMap; // object colors per fragment
uniform sampler2D normalMapR; // normals per vertex
uniform sampler2D normalMapG; // normals per vertex
uniform sampler2D normalMapB; // normals per vertex

uniform float ambientCoeff;
uniform float shininess;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 viewPos;

void main()
{
    // obtain normal map from texture [0,1]
    vec3 normalr = texture(normalMapR, TexCoords).rgb;

    // transform vector to range [-1,1]
    normalr = normalize((normalr * 2.0) - 1.0);

    // get diffuse color
    vec3 color = texture(diffuseMap, TexCoords).rgb;

    // ambient color for object
    vec3 ambientColor = color * ambientCoeff;

    // costheta
    vec3 lightDir = normalize(TangentLightPos - TangentFragPos);
    float costhetaR = dot(lightDir, normalr);
    costhetaR = max(costhetaR, 0.0);
    vec3 diffuseColor = vec3(1.0);
    diffuseColor = color * costhetaR;

    // specular
    vec3 viewDir = normalize(TangentViewPos - TangentFragPos);
    // vec3 reflectDir = reflect(-lightDir, normal);
    vec3 halfway = normalize(lightDir + viewDir);
    float specAngleR = dot(normalr, halfway);
    specAngleR = max(specAngleR, 0.0);
    specAngleR = pow(specAngleR, shininess);
    vec3 specularColor = vec3(1.0);
    specularColor = lightColor * specAngleR;

    // final fragment color
    FragColor = vec4(ambientColor + diffuseColor + specularColor, 1.0);
}
"""


lampVshader = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

lampFshader = """
#version 330 core
out vec4 FragColor;

uniform lowp vec3 lightColor;

void main()
{
    FragColor = vec4(lightColor, 1.0);
}
"""

rgbptmVshader = """
#version 330 core
layout (location = 0) in vec3 aPos;

layout (location = 1) in float acoeff1r;
layout (location = 2) in float acoeff2r;
layout (location = 3) in float acoeff3r;
layout (location = 4) in float acoeff4r;
layout (location = 5) in float acoeff5r;
layout (location = 6) in float acoeff6r;

layout (location = 7) in float acoeff1g;
layout (location = 8) in float acoeff2g;
layout (location = 9) in float acoeff3g;
layout (location = 10) in float acoeff4g;
layout (location = 11) in float acoeff5g;
layout (location = 12) in float acoeff6g;

layout (location = 13) in float acoeff1b;
layout (location = 14) in float acoeff2b;
layout (location = 15) in float acoeff3b;
layout (location = 16) in float acoeff4b;
layout (location = 17) in float acoeff5b;
layout (location = 18) in float acoeff6b;

// six coeff per channel
out float coeff1r;
out float coeff2r;
out float coeff3r;
out float coeff4r;
out float coeff5r;
out float coeff6r;

out float coeff1g;
out float coeff2g;
out float coeff3g;
out float coeff4g;
out float coeff5g;
out float coeff6g;

out float coeff1b;
out float coeff2b;
out float coeff3b;
out float coeff4b;
out float coeff5b;
out float coeff6b;

out vec3 FragPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    gl_Position = projection * view * model * vec4(aPos, 1.0);

    coeff1r = acoeff1r;
    coeff2r = acoeff2r;
    coeff3r = acoeff3r;
    coeff4r = acoeff4r;
    coeff5r = acoeff5r;
    coeff6r = acoeff6r;

    coeff1g = acoeff1g;
    coeff2g = acoeff2g;
    coeff3g = acoeff3g;
    coeff4g = acoeff4g;
    coeff5g = acoeff5g;
    coeff6g = acoeff6g;

    coeff1b = acoeff1b;
    coeff2b = acoeff2b;
    coeff3b = acoeff3b;
    coeff4b = acoeff4b;
    coeff5b = acoeff5b;
    coeff6b = acoeff6b;
}
"""

rgbptmFshader = """
#version 330 core

out vec4 FragColor;

in vec3 FragPos;

// six coeff per channel

in float coeff1r;
in float coeff2r;
in float coeff3r;
in float coeff4r;
in float coeff5r;
in float coeff6r;

in float coeff1g;
in float coeff2g;
in float coeff3g;
in float coeff4g;
in float coeff5g;
in float coeff6g;

in float coeff1b;
in float coeff2b;
in float coeff3b;
in float coeff4b;
in float coeff5b;
in float coeff6b;

uniform vec3 lightPos;
uniform vec3 viewPos;

uniform vec3 lightColor;
uniform float shininess;

uniform vec3 attc;

uniform vec3 diffuseCoeffs;
uniform vec3 specularCoeffs;
uniform vec3 ambientCoeffs;

uniform bool blinn;

float computeLuPerChannel(float c0, float c1, float c2, float c3, float c4);
float computeLvPerChannel(float c0, float c1, float c2, float c3, float c4);
vec3 computeNormalPerChannel(float c0, float c1, float c2, float c3, float c4);
float computeDiffuseColorPerChannel(float c0, float c1, float c2, float c3,
                                    float c4, float c5, float Lu, float Lv);
vec3 computeDiffuseColor(float c0r, float c1r,
                         float c2r, float c3r,
                         float c4r, float c5r,
                         float LuR, float LvR,
                         float c0g, float c1g,
                         float c2g, float c3g,
                         float c4g, float c5g,
                         float LuG, float LvG,
                         float c0b, float c1b,
                         float c2b, float c3b,
                         float c4b, float c5b,
                         float LuB, float LvB,
                         );

float computeAttenuation(float attC1, float attC2, float attC3, float dist);

float computeFragColorPerChannel(vec3 surfaceNormal, float attc1, 
                                float attc2, float attc3,
                                float objDiffuseChannelIntensity,
                                float objDiffuseChannelCoeff,
                                float ambientChannelCoeff,
                                float ambientChannelIntensity,
                                float specularChannelIntensity,
                                float specularChannelCoeff,
                                float lightChannelIntensity,
                                float shininess, vec3 viewerPosition,
                                vec3 fragPos, vec3 lightPosition,
                                bool blinn);

void main()
{
    vec3 surfaceNormalR = computeNormalPerChannel(
        coeff1r, coeff2r, coeff3r, coeff4r, coeff5r
    );
    vec3 surfaceNormalG = computeNormalPerChannel(
        coeff1g, coeff2g, coeff3g, coeff4g, coeff5g
    );
    vec3 surfaceNormalB = computeNormalPerChannel(
        coeff1b, coeff2b, coeff3b, coeff4b, coeff5b
    );
    vec3 diffuseColor = computeDiffuseColor(
        coeff1r, coeff2r, coeff3r, coeff4r, coeff5r, coeff6r,
        coeff1g, coeff2g, coeff3g, coeff4g, coeff5g, coeff6g,
        coeff1b, coeff2b, coeff3b, coeff4b, coeff5b, coeff6b,
    );

    float redC = computeFragColorPerChannel(
                    surfaceNormalR,
                    attc.x,
                    attc.y, attc.z,
                    diffuseColor.x,
                    diffuseCoeffs.x,
                    ambientCoeffs.x,
                    lightColor.x,  // ambient channel intensity
                    diffuseColor.x,  // specular channel intensity
                    specularCoeffs.x,
                    lightColor.x,
                    shininess,
                    viewPos,
                    FragPos,
                    lightPos,
                    blinn);
    float greenC = computeFragColorPerChannel(
                    surfaceNormalG,
                    attc.x,
                    attc.y, attc.z,
                    diffuseColor.y,
                    diffuseCoeffs.y,
                    ambientCoeffs.y,
                    lightColor.y,  // ambient channel intensity
                    diffuseColor.y,  // specular channel intensity
                    specularCoeffs.y,
                    lightColor.y,  // light channel intensity
                    shininess,
                    viewPos,
                    FragPos,
                    lightPos,
                    blinn);
    float blueC = computeFragColorPerChannel(
                    surfaceNormalB,
                    attc.x,
                    attc.y, attc.z,
                    diffuseColor.z,
                    diffuseCoeffs.z,
                    ambientCoeffs.z,
                    lightColor.z,  // ambient channel intensity
                    diffuseColor.z,  // specular channel intensity
                    specularCoeffs.z,
                    lightColor.z,  // light channel intensity
                    shininess,
                    viewPos,
                    FragPos,
                    lightPos,
                    blinn);
    FragColor = vec4(redC, greenC, blueC, 1.0);
}

float computeLuPerChannel(float c0, float c1, float c2, float c3, float c4)
{
    // taken directly from the paper of Tom Malzbender, Dan Gelb, Hans Wolters
    // http://www.hpl.hp.com/ptm
    return ((c2 * c4) - (2 * c1 * c3)) / ((4 * c0 * c1) - (c2 * c2));
}

float computeLvPerChannel(float c0, float c1, float c2, float c3, float c4)
{ 
    // taken directly from the paper of Tom Malzbender, Dan Gelb, Hans Wolters
    // http://www.hpl.hp.com/ptm
    return ((c2 * c3) - (2 * c0 * c4)) / ((4 * c0 * c1) - (c2 * c2));
}

vec3 computeNormalPerChannel(float c0, float c1, float c2, float c3, float c4)
{
    // taken directly from the paper of Tom Malzbender, Dan Gelb, Hans Wolters
    // http://www.hpl.hp.com/ptm
    float Lu = computeLuPerChannel(float c0, float c1,
                                   float c2, float c3,
                                   float c4);
    float Lv = computeLvPerChannel(float c0, float c1,
                                   float c2, float c3,
                                   float c4);

    return vec3(Lu, Lv, sqrt(1 - (Lu * Lu) - (Lv * Lv));
}

float computeDiffuseColorPerChannel(float c0, float c1, float c2, float c3,
                         float c4, float c5, float Lu, float Lv)
{
    // taken directly from the paper of Tom Malzbender, Dan Gelb, Hans Wolters
    // http://www.hpl.hp.com/ptm
    float term1 = c0 * Lu * Lu;
    float term2 = c1 * Lv * Lv;
    float term3 = c2 * Lu * Lv;
    float term4 = c3 * Lu;
    float term5 = c4 * Lv;
    float term6 = c5;
    return term1 + term2 + term3 + term4 + term5 + term6;
}
vec3 computeDiffuseColor(float c0r, float c1r,
                         float c2r, float c3r,
                         float c4r, float c5r,
                         float c0g, float c1g,
                         float c2g, float c3g,
                         float c4g, float c5g,
                         float c0b, float c1b,
                         float c2b, float c3b,
                         float c4b, float c5b)
{
    float LuR = computeLuPerChannel(
        c1r, c2r, c3r, c4r, c5r, c6r
    );
    float LvR = computeLvPerChannel(
        c1r, c2r, c3r, c4r, c5r, c6r
    );
    float LuG = computeLuPerChannel(
        c1g, c2g, c3g, c4g, c5g, c6g
    );
    float LvG = computeLvPerChannel(
        c1g, c2g, c3g, c4g, c5g, c6g
    );
    float LuB = computeLuPerChannel(
        c1b, c2b, c3b, c4b, c5b, c6b
    );
    float LvB = computeLvPerChannel(
        c1b, c2b, c3b, c4b, c5b, c6b
    );
    return vec3(
    computeDiffuseColorPerChannel(c0r, c1r, c2r, c3r, c4r, c5r, LuR, LvR),
    computeDiffuseColorPerChannel(c0g, c1g, c2g, c3g, c4g, c5g, LuG, LvG),
    computeDiffuseColorPerChannel(c0b, c1b, c2b, c3b, c4b, c5b, LuB, LvB)
    );
}

float computeAttenuation(float attC1, float attC2, float attC3, float dist)
{
    // f_att = min(\frac{1}{c_1 + c_2{\times}d_L + c_3{\times}d^2_{L}} , 1)
    float distSqr = dist * dist;
    float att1 = dist * attC2;
    float att2 = distSqr * attC3;
    float result = attC1 + att2 + att1;
    result = 1 / result;
    return min(result, 1);
}

float computeFragColorPerChannel(vec3 surfaceNormal, float attc1, 
                                float attc2, float attc3,
                                float objDiffuseChannelIntensity,
                                float objDiffuseChannelCoeff,
                                float ambientChannelCoeff,
                                float ambientChannelIntensity,
                                float specularChannelIntensity,
                                float specularChannelCoeff,
                                float lightChannelIntensity,
                                float shininess, vec3 viewerPosition,
                                vec3 fragPos, vec3 lightPosition,
                                bool blinn)
{
    // blinn phong light illumination
    float ambientTerm = ambientChannelIntensity * ambientChannelCoeff;
    ambientTerm = ambientTerm * objDiffuseChannelIntensity;

    // attenuation
    float dist = distance(lightPosition, fragPos);
    float fattr = computeAttenuation(attc1, attc2, attc3, dist);

    // lambertian term
    vec3 snormal = normalize(surfaceNormal);
    vec3 lightDirection = normalize(lightPosition - fragPos);
    float costheta = dot(lightDirection, snormal);
    float lambertian = costheta * objDiffuseChannelCoeff;
    lambertian = lambertian * objDiffuseChannelIntensity;
    lambertian = lambertian * lightChannelIntensity;
    lambertian = lambertian * fattr;

    // specular term
    vec3 viewerDirection = normalize(viewerPosition - fragPos);
    float specAngle = 0.0;
    if (blinn)
    {
        vec3 halfway = normalize(lightDirection + viewerDirection);
        specAngle = max(dot(snormal, halfway), 0);
    }else{
        vec3 reflectionDir = reflect(-lightDirection, snormal);
        specAngle = max(dot(viewerDirection, reflectionDir), 0);
    }
    float specTerm = specularChannelIntensity * specularChannelCoeff;
    specTerm = pow(specAngle, shininess) * specTerm;
    return ambientTerm + lambertian + specTerm;
}

"""

shaders = {
    "quad": {
        "fragment": quadFshader,
        "vertex": quadVshader,
        "attribute_info": {
            "stride": None,
            "aPos": {"layout": 0, "size": 3, "offset": 0},
            "aNormal": {"layout": 1, "size": 3, "offset": 3},
            "aTexCoord": {"layout": 2, "size": 2, "offset": 6},
            "aTangent": {"layout": 3, "size": 3, "offset": 8},
            "aBiTangent": {"layout": 4, "size": 3, "offset": 11},
        },
    },
    "lambert": {
        "fragment": lambertFshader,
        "vertex": lambertVshader,
        "attribute_info": {
            "stride": None,
            "aPos": {"layout": 0, "size": 3, "offset": 0},
            "aNormal": {"layout": 1, "size": 3, "offset": 3},
            "aTexCoord": {"layout": 2, "size": 2, "offset": 6},
        },
    },
    "quadPerChannel": {
        "fragment": quadFshaderPerChannel,
        "vertex": quadVshader,
        "attribute_info": {
            "stride": None,
            "aPos": {"layout": 0, "size": 3, "offset": 0},
            "aNormal": {"layout": 1, "size": 3, "offset": 3},
            "aTexCoord": {"layout": 2, "size": 2, "offset": 6},
            "aTangent": {"layout": 3, "size": 3, "offset": 8},
            "aBiTangent": {"layout": 4, "size": 3, "offset": 11},
        },
    },
    "quadPerChannelTest": {
        "fragment": quadFshaderPerChannelTest,
        "vertex": quadVshader,
        "attribute_info": {
            "stride": None,
            "aPos": {"layout": 0, "size": 3, "offset": 0},
            "aNormal": {"layout": 1, "size": 3, "offset": 3},
            "aTexCoord": {"layout": 2, "size": 2, "offset": 6},
            "aTangent": {"layout": 3, "size": 3, "offset": 8},
            "aBiTangent": {"layout": 4, "size": 3, "offset": 11},
        },
    },
    "lamp": {
        "fragment": lampFshader,
        "vertex": lampVshader,
        "attribute_info": {
            "stride": None,
            "aPos": {"layout": 0, "size": 3, "offset": 0},
        },
    },
    "rgbptm": {
        "fragment": rgbptmFshader,
        "vertex": rgbptmVshader,
        "attribute_info": {
            "stride": None,
            "aPos": {"layout": 0, "size": 3, "offset": 0},
            "acoeff1r": {"layout": 1, "size": 1, "offset": 3},
            "acoeff2r": {"layout": 2, "size": 1, "offset": 4},
            "acoeff3r": {"layout": 3, "size": 1, "offset": 5},
            "acoeff4r": {"layout": 4, "size": 1, "offset": 6},
            "acoeff5r": {"layout": 5, "size": 1, "offset": 7},
            "acoeff6r": {"layout": 6, "size": 1, "offset": 8},
            "acoeff1g": {"layout": 7, "size": 1, "offset": 9},
            "acoeff2g": {"layout": 8, "size": 1, "offset": 10},
            "acoeff3g": {"layout": 9, "size": 1, "offset": 11},
            "acoeff4g": {"layout": 10, "size": 1, "offset": 12},
            "acoeff5g": {"layout": 11, "size": 1, "offset": 13},
            "acoeff6g": {"layout": 12, "size": 1, "offset": 14},
            "acoeff1b": {"layout": 13, "size": 1, "offset": 15},
            "acoeff2b": {"layout": 14, "size": 1, "offset": 16},
            "acoeff3b": {"layout": 15, "size": 1, "offset": 17},
            "acoeff4b": {"layout": 16, "size": 1, "offset": 18},
            "acoeff5b": {"layout": 17, "size": 1, "offset": 19},
            "acoeff6b": {"layout": 18, "size": 1, "offset": 20},
        },
    },
}
