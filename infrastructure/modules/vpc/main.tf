# infrastructure/modules/vpc/main.tf
resource "vkcs_networking_network" "main" {
  name = var.network_name
}

resource "vkcs_networking_subnet" "main" {
  name       = "${var.network_name}-subnet"
  network_id = vkcs_networking_network.main.id
  cidr       = var.subnet_cidr
  ip_version = 4
}

resource "vkcs_networking_router" "main" {
  name                = "${var.network_group_name}-router"
  external_network_id = var.external_network_id
}

resource "vkcs_networking_router_interface" "main" {
  router_id = vkcs_networking_router.main.id
  subnet_id = vkcs_networking_subnet.main.id
}

# Security Group для нод Kubernetes
resource "vkcs_networking_secgroup" "nodes" {
  name        = "${var.network_group_name}-nodes-sg"
  description = "Security group for Kubernetes nodes"
}

resource "vkcs_networking_secgroup_rule" "nodes_ssh" {
  direction         = "ingress"
  ethertype         = "IPv4"
  secgroup_id       = vkcs_networking_secgroup.nodes.id
  port_range_min    = 22
  port_range_max    = 22
  protocol          = "tcp"
  remote_ip_prefix  = "0.0.0.0/0"
}

resource "vkcs_networking_secgroup_rule" "nodes_k8s" {
  direction         = "ingress"
  ethertype         = "IPv4"
  secgroup_of_group = true
  secgroup_id       = vkcs_networking_secgroup.nodes.id
  protocol          = "tcp"
  port_range_min    = 0
  port_range_max    = 65535
}

# Security Group для Ingress (внешний трафик)
resource "vkcs_networking_secgroup" "ingress" {
  name        = "${var.network_group_name}-ingress-sg"
  description = "Security group for ingress traffic"
}

resource "vkcs_networking_secgroup_rule" "ingress_http" {
  direction         = "ingress"
  ethertype         = "IPv4"
  secgroup_id       = vkcs_networking_secgroup.ingress.id
  port_range_min    = 80
  port_range_max    = 80
  protocol          = "tcp"
  remote_ip_prefix  = "0.0.0.0/0"
}

resource "vkcs_networking_secgroup_rule" "ingress_https" {
  direction         = "ingress"
  ethertype         = "IPv4"
  secgroup_id       = vkcs_networking_secgroup.ingress.id
  port_range_min    = 443
  port_range_max    = 443
  protocol          = "tcp"
  remote_ip_prefix  = "0.0.0.0/0"
}

output "network_id" { value = vkcs_networking_network.main.id }
output "subnet_id" { value = vkcs_networking_subnet.main.id }
output "router_id" { value = vkcs_networking_router.main.id }
output "nodes_secgroup_id" { value = vkcs_networking_secgroup.nodes.id }
output "ingress_secgroup_id" { value = vkcs_networking_secgroup.ingress.id }