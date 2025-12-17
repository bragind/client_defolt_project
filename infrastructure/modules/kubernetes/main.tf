# infrastructure/modules/kubernetes/main.tf
resource "vkcs_kubernetes_cluster" "ml" {
  name                = var.cluster_name
  cluster_template_id = var.cluster_template_id
  master_count        = var.master_count
  master_flavor       = var.master_flavor
  network_id          = var.network_id
  subnet_id           = var.subnet_id
  security_group_ids  = [var.nodes_secgroup_id]
  availability_zone   = var.availability_zone
}

# CPU Node Pool
resource "vkcs_kubernetes_node_group" "cpu" {
  cluster_id      = vkcs_kubernetes_cluster.ml.id
  name            = "cpu-pool"
  flavor          = var.cpu_flavor
  count           = var.cpu_node_count
  max_nodes       = var.cpu_max_nodes
  min_nodes       = var.cpu_min_nodes
  security_groups = [var.nodes_secgroup_id]
}

# GPU Node Pool (опционально)
resource "vkcs_kubernetes_node_group" "gpu" {
  count           = var.enable_gpu ? 1 : 0
  cluster_id      = vkcs_kubernetes_cluster.ml.id
  name            = "gpu-pool"
  flavor          = var.gpu_flavor
  count           = var.gpu_node_count
  max_nodes       = var.gpu_max_nodes
  min_nodes       = var.gpu_min_nodes
  security_groups = [var.nodes_secgroup_id]
}

# Outputs для kubectl
output "cluster_host" { value = vkcs_kubernetes_cluster.ml.host }
output "cluster_token" { value = vkcs_kubernetes_cluster.ml.token }
output "cluster_ca_cert" { value = vkcs_kubernetes_cluster.ml.ca_cert }
output "cluster_id" { value = vkcs_kubernetes_cluster.ml.id }