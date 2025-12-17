# infrastructure/terraform/providers.tf
terraform {
  required_version = ">= 1.5"
  required_providers {
    vkcs = {
      source  = "vk-cs/vkcs"
      version = "~> 0.12"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

provider "vkcs" {
  auth_url    = "https://api.selvpc.ru/identity/v3"
  username    = var.username
  password    = var.password
  project_id  = var.project_id
  user_domain_name = "users"
  project_domain_name = "users"
  region      = var.region
}

provider "kubernetes" {
  host                   = module.kubernetes.cluster_host
  token                  = module.kubernetes.cluster_token
  cluster_ca_certificate = base64decode(module.kubernetes.cluster_ca_cert)
}