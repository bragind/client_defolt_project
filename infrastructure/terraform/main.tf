# infrastructure/terraform/main.tf
module "vpc" {
  source = "../modules/vpc"

  network_name        = var.network_name
  subnet_cidr         = var.subnet_cidr
  external_network_id = var.external_network_id
  network_group_name  = var.network_group_name
}

module "kubernetes" {
  source = "../modules/kubernetes"

  cluster_name        = var.cluster_name
  cluster_template_id = var.cluster_template_id
  master_flavor       = var.master_flavor
  cpu_flavor          = var.cpu_flavor
  gpu_flavor          = var.gpu_flavor
  enable_gpu          = var.enable_gpu

  network_id         = module.vpc.network_id
  subnet_id          = module.vpc.subnet_id
  nodes_secgroup_id  = module.vpc.nodes_secgroup_id
  availability_zone  = var.availability_zone

  cpu_node_count = var.cpu_node_count
  cpu_max_nodes  = var.cpu_max_nodes
  cpu_min_nodes  = var.cpu_min_nodes
  gpu_node_count = var.gpu_node_count
  gpu_max_nodes  = var.gpu_max_nodes
  gpu_min_nodes  = var.gpu_min_nodes
}

module "storage" {
  source = "../modules/storage"
  # ... настройки Object Storage для state и моделей
}