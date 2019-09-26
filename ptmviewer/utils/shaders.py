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
    float distanceLightFrag = length(TangentLightPos - TangentFragPos);
    float att = computeAttenuation(attc1, attc2, attc3, distanceLightFrag);

    // ambient color for object
    vec3 ambientColor = color * ambientCoeff;


    // simple light direction
    vec3 lightDir = normalize(TangentLightPos - TangentFragPos);

    // spotlight cone theta
    float theta = dot(lightDir, lightDirection);
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
    return min(result, 1.0);
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

layout (location = 1) in vec3 acoeff13r;
layout (location = 2) in vec3 acoeff46r;

layout (location = 3) in vec3 acoeff13g;
layout (location = 4) in vec3 acoeff46g;

layout (location = 5) in vec3 acoeff13b;
layout (location = 6) in vec3 acoeff46b;

// six coeff per channel

out vec3 DiffColor;
out vec3 NormalR;
out vec3 NormalG;
out vec3 NormalB;
out vec3 FragPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

float computeLuPerChannel(float c0, float c1, float c2, float c3, float c4);
float computeLvPerChannel(float c0, float c1, float c2, float c3, float c4);
vec3 computeNormalPerChannel(float c0, float c1, float c2, float c3, float c4);
float computeDiffuseColorPerChannel(float c0, float c1, float c2, float c3,
                                    float c4, float c5, float Lu, float Lv);

vec3 computeDiffuseColor(float c0r, float c1r,
                         float c2r, float c3r,
                         float c4r, float c5r,
                         float c0g, float c1g,
                         float c2g, float c3g,
                         float c4g, float c5g,
                         float c0b, float c1b,
                         float c2b, float c3b,
                         float c4b, float c5b);

void main()
{
    vec3 surfaceNormalR = computeNormalPerChannel(
        coeff13r.x, coeff13r.y, coeff13r.z, coeff46r.x, coeff46r.y
    );
    vec3 sugfaceNormalG = computeNormalPerChannel(
        coeff13g.x, coeff13g.y, coeff13g.z, coeff46g.x, coeff46g.y
    );
    vec3 surfaceNormalB = computeNormalPerChannel(
        coeff13b.x, coeff13b.y, coeff13b.z, coeff46b.x, coeff46b.y
    );
    vec3 diffColor = computeDiffuseColor(
        coeff13r.x, coeff13r.y, coeff13r.z, coeff46r.x, coeff46r.y, coeff46r.z,
        coeff13g.x, coeff13g.y, coeff13g.z, coeff46g.x, coeff46g.y, coeff46g.z,
        coeff13b.x, coeff13b.y, coeff13b.z, coeff46b.x, coeff46b.y, coeff46b.z
        );
    DiffColor = diffColor;
    NormalR = surfaceNormalR;
    NormalG = surfaceNormalG;
    NormalB = surfaceNormalB;
    
    FragPos = vec3(model * vec4(aPos, 1.0));
    gl_Position = projection * view * model * vec4(aPos, 1.0);
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
    float Lu = computeLuPerChannel(c0,c1,
                                   c2, c3,
                                   c4);
    float Lv = computeLvPerChannel(c0, c1,
                                   c2, c3,
                                   c4);

    return vec3(Lu, Lv, sqrt(1 - (Lu * Lu) - (Lv * Lv)));
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
        c0r, c1r, c2r, c3r, c4r
    );
    float LvR = computeLvPerChannel(
        c0r, c1r, c2r, c3r, c4r
    );
    float LuG = computeLuPerChannel(
        c0g, c1g, c2g, c3g, c4g
    );
    float LvG = computeLvPerChannel(
        c0g, c1g, c2g, c3g, c4g
    );
    float LuB = computeLuPerChannel(
        c0b, c1b, c2b, c3b, c4b
    );
    float LvB = computeLvPerChannel(
        c0b, c1b, c2b, c3b, c4b
    );
    return vec3(
    computeDiffuseColorPerChannel(c0r, c1r, c2r, c3r, c4r, c5r, LuR, LvR),
    computeDiffuseColorPerChannel(c0g, c1g, c2g, c3g, c4g, c5g, LuG, LvG),
    computeDiffuseColorPerChannel(c0b, c1b, c2b, c3b, c4b, c5b, LuB, LvB)
    );
}


"""

rgbptmFshader = """
#version 330 core

out vec4 FragColor;

in vec3 FragPos;

// six coeff per channel

in vec3 NormalR;
in vec3 NormalG;
in vec3 NormalB;

in vec3 DiffColor;

uniform vec3 lightPos;
uniform vec3 viewPos;

uniform vec3 lightColor;
uniform float shininess;

uniform vec3 attc;

uniform vec3 diffuseCoeffs;
uniform vec3 specularCoeffs;
uniform vec3 ambientCoeffs;

uniform bool blinn;

float computeAttenuation(float attC1, float attC2, float attC3, float distVal);

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
        vec3 diffuseColor = computeDiffuseColor(coeff1r, coeff2r, coeff3r, 
                                            coeff4r, coeff5r, coeff6r, 
                                            coeff1g, coeff2g, coeff3g, 
                                            coeff4g, coeff5g, coeff6g,
                                            coeff1b, coeff2b, coeff3b,
                                            coeff4b, coeff5b, coeff6b);

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
    "lamp": {
        "fragment": lampFshader,
        "vertex": lampVshader,
        "attribute_info": {
            "stride": None,
            "aPos": {"layout": 0, "size": 3, "offset": 0},
        },
    },
    "rgbcoeff": {
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
