# author: Kaan Eraslan
# license: see, LICENSE.
# Simple file for holding shaders

lambertVshader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = aNormal;
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

objectVshader = """
#version 330 core

// specify input
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aBiTangent;

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

objectFshader = """
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

objectFshaderPerChannel = """
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
    vec3 normalg = texture(normalMapG, TexCoords).rgb;
    vec3 normalb = texture(normalMapB, TexCoords).rgb;

    // transform vector to range [-1,1]
    normalr = normalize((normalr * 2.0) - 1.0);
    normalg = normalize((normalg * 2.0) - 1.0);
    normalb = normalize((normalb * 2.0) - 1.0);

    // get diffuse color
    vec3 color = texture(diffuseMap, TexCoords).rgb;

    // ambient color for object
    vec3 ambientColor = color * ambientCoeff;

    // costheta
    vec3 lightDir = normalize(TangentLightPos - TangentFragPos);
    float costhetaR = dot(lightDir, normalr);
    float costhetaG = dot(lightDir, normalg);
    float costhetaB = dot(lightDir, normalb);
    costhetaR = max(costhetaR, 0.0);
    costhetaG = max(costhetaG, 0.0);
    costhetaB = max(costhetaB, 0.0);
    vec3 diffuseColor = vec3(1.0);
    diffuseColor.x = color.x * costhetaR;
    diffuseColor.y = color.y * costhetaG;
    diffuseColor.z = color.z * costhetaB;

    // specular
    vec3 viewDir = normalize(TangentViewPos - TangentFragPos);
    // vec3 reflectDir = reflect(-lightDir, normal);
    vec3 halfway = normalize(lightDir + viewDir);
    float specAngleR = dot(normalr, halfway);
    float specAngleG = dot(normalg, halfway);
    float specAngleB = dot(normalb, halfway);
    specAngleR = max(specAngleR, 0.0);
    specAngleG = max(specAngleG, 0.0);
    specAngleB = max(specAngleB, 0.0);
    specAngleR = pow(specAngleR, shininess);
    specAngleG = pow(specAngleG, shininess);
    specAngleB = pow(specAngleB, shininess);
    vec3 specularColor = vec3(1.0);
    specularColor.x = lightColor.x * specAngleR;
    specularColor.y = lightColor.y * specAngleG;
    specularColor.z = lightColor.z * specAngleB;

    // final fragment color
    FragColor = vec4(ambientColor + diffuseColor + specularColor, 1.0);
}
"""
objectFshaderPerChannelTest = """
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

layout (location = 1) in float acoeff1g;
layout (location = 2) in float acoeff2g;
layout (location = 3) in float acoeff3g;
layout (location = 4) in float acoeff4g;
layout (location = 5) in float acoeff5g;
layout (location = 6) in float acoeff6g;

layout (location = 1) in float acoeff1b;
layout (location = 2) in float acoeff2b;
layout (location = 3) in float acoeff3b;
layout (location = 4) in float acoeff4b;
layout (location = 5) in float acoeff5b;
layout (location = 6) in float acoeff6b;

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

uniform float coeffRed;
uniform float coeffGreen;
uniform float coeffBlue;

void main()
{
    vec3 lightDirection = normalize(lightPos - FragPos);
    float lu = lightDirection.x;
    float lv = lightDirection.y;

    float c1r = lu * lu * coeff1r;
    float c2r = lv * lv * coeff2r;
    float c3r = lu * lv * coeff3r;
    float c4r = lu * coeff4r;
    float c5r = lv * coeff5r;
    float c6r = coeff6r;
    float cr = c1r + c2r + c3r + c4r + c5r + c6r;

    float c1g = lu * lu * coeff1g;
    float c2g = lv * lv * coeff2g;
    float c3g = lu * lv * coeff3g;
    float c4g = lu * coeff4g;
    float c5g = lv * coeff5g;
    float c6g = coeff6g;
    float cg = c1g + c2g + c3g + c4g + c5g + c6g;

    float c1b = lu * lu * coeff1b;
    float c2b = lv * lv * coeff2b;
    float c3b = lu * lv * coeff3b;
    float c4b = lu * coeff4b;
    float c5b = lv * coeff5b;
    float c6b = coeff6b;
    float cb = c1b + c2b + c3b + c4b + c5b + c6b;
    FragColor = vec4(cr * coeffRed, cg * coeffGreen, cb * coeffBlue, 1.0);
}
"""

shaders = {
    "quad": {"fragment": objectFshader, "vertex": objectVshader},
    "quadPerChannel": {
        "fragment": objectFshaderPerChannel,
        "vertex": objectVshader,
    },
    "quadPerChannelTest": {
        "fragment": objectFshaderPerChannelTest,
        "vertex": objectVshader,
    },
    "lamp": {"fragment": lampFshader, "vertex": lampVshader},
    "rgbptm": {"fragment": rgbptmFshader, "vertex": rgbptmVshader},
}
