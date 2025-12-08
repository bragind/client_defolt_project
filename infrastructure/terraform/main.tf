terraform {
  required_version = ">= 1.0"
  
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.84.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.16.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.8.0"
    }
  }
  
  backend "s3" {
    endpoint   = "storage.yandexcloud.net"
    bucket     = "terraform-state-credit-scoring"
    region     = "ru-central1"
    key        = "terraform.tfstate"
    access_key = var.yc_access_key
    secret_key = var.yc_secret_key
    
    skip_region_validation      = true
    skip_credentials_validation = true
  }
}

provider "yandex" {
  token     = var.yc_token
  cloud_id  = var.yc_cloud_id
  folder_id = var.yc_folder_id
  zone      = "ru-central1-a"
}

# Модуль VPC
module "vpc" {
  source = "./modules/vpc"
  
  vpc_name           = "credit-scoring-vpc"
  vpc_description    = "VPC for Credit Scoring System"
  subnet_cidr_blocks = ["10.0.1.0/24"]
  zones              = ["ru-central1-a"]
}

# Модуль Managed Kubernetes
module "kubernetes" {
  source = "./modules/kubernetes"
  
  cluster_name        = "credit-scoring-k8s"
  cluster_description = "Kubernetes cluster for Credit Scoring"
  network_id         = module.vpc.network_id
  subnet_id          = module.vpc.subnet_ids[0]
  service_account_id = yandex_iam_service_account.k8s.id
  
  node_groups = {
    "cpu-pool" = {
      cores         = 4
      memory        = 8
      disk_size     = 50
      node_count    = 2
      preemptible   = false
      auto_scale    = true
      min_size      = 2
      max_size      = 5
    }
    "gpu-pool" = {
      cores         = 8
      memory        = 32
      disk_size     = 100
      node_count    = 1
      preemptible   = true
      gpu_count     = 1
      gpu_type      = "gpu-standard-v3"
      auto_scale    = false
    }
  }
}

# Модуль хранилища
module "storage" {
  source = "./modules/storage"
  
  bucket_name     = "credit-scoring-models"
  bucket_access   = "private"
  use_versioning  = true
  encrypt_bucket  = true
}

# Модуль мониторинга
module "monitoring" {
  source = "./modules/monitoring"
  
  prometheus_enabled = true
  grafana_enabled    = true
  alertmanager_enabled = true
  loki_enabled       = true
  tempo_enabled      = true
}