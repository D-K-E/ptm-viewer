#version 420 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;

// material struct

struct Material {
    vec3 ambientColor;
    vec3 diffuseColor;
    vec3 specularColor;
    float shininess;
};

uniform Material material;

struct Light {
    vec3 position;

    vec3 ambientColor;
    vec3 diffuseColor;
    vec3 specularColor;
};

uniform Light light;

struct Coeffs {
    float ambient;
    float diffuse;
    float specular;
};

uniform Coeffs coeffs;

uniform vec3 viewerPosition;

void main() {
    // normalize normals
    vec3 norm = normalize(Normal);
    vec3 lightDirection = normalize(light.position - FragPos);
    float costheta = max(dot(norm, lightDirection), 0.0);

    // diffuse color
    vec3 diffuseColor = light.diffuseColor * material.diffuseColor;
    diffuseColor = diffuseColor * coeffs.diffuse * costheta;

    // ambient term
    vec3 ambientTerm = light.ambientColor * material.ambientColor;
    ambientTerm = ambientTerm * coeffs.ambient;

    // specular color
    vec3 viewerDirection = normalize(viewerPosition - FragPos);
    vec3 reflectionDirection = reflect(-lightDirection, norm);
    float specularAngle = max(dot(viewerDirection, reflectionDirection), 0.0);
    specularAngle = pow(specularAngle, material.shininess);
    vec3 specular = light.specularColor * material.specularColor;
    specular = specular * specularAngle * coeffs.specular;

    vec3 result = specular + ambientTerm + diffuseColor;
    FragColor = vec4(result, 1.0);
}
