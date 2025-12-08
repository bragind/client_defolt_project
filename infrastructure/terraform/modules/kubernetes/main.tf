resource "yandex_kubernetes_cluster" "cluster" {
  name        = var.cluster_name
  description = var.cluster_description
  network_id  = var.network_id
  
  master {
    version   = "1.24"
    public_ip = true
    
    master_location {
      zone      = var.zone
      subnet_id = var.subnet_id
    }
    
    maintenance_policy {
      auto_upgrade = true
      
      maintenance_window {
        start_time = "23:00"
        duration   = "3h"
      }
    }
    
    security_group_ids = [yandex_vpc_security_group.k8s_master.id]
  }
  
  service_account_id      = var.service_account_id
  node_service_account_id = var.service_account_id
  
  kms_provider {
    key_id = yandex_kms_symmetric_key.k8s_key.id
  }
  
  labels = var.labels
}

resource "yandex_kubernetes_node_group" "node_groups" {
  for_each = var.node_groups
  
  cluster_id  = yandex_kubernetes_cluster.cluster.id
  name        = "${var.cluster_name}-${each.key}"
  description = "Node group ${each.key} for ${var.cluster_name}"
  
  instance_template {
    platform_id = "standard-v3"
    
    resources {
      memory = each.value.memory * 1024
      cores  = each.value.cores
      
      dynamic "gpus" {
        for_each = each.value.gpu_count != null ? [1] : []
        content {
          count = each.value.gpu_count
          gpu_cluster_id = each.value.gpu_type
        }
      }
    }
    
    boot_disk {
      type = "network-ssd"
      size = each.value.disk_size
    }
    
    scheduling_policy {
      preemptible = each.value.preemptible
    }
    
    network_interface {
      subnet_ids = [var.subnet_id]
      nat        = true
    }
    
    metadata = {
      ssh-keys = "ubuntu:${file("~/.ssh/id_rsa.pub")}"
    }
  }
  
  scale_policy {
    dynamic "auto_scale" {
      for_each = each.value.auto_scale ? [1] : []
      content {
        min     = each.value.min_size
        max     = each.value.max_size
        initial = each.value.node_count
      }
    }
    
    dynamic "fixed_scale" {
      for_each = !each.value.auto_scale ? [1] : []
      content {
        size = each.value.node_count
      }
    }
  }
  
  allocation_policy {
    location {
      zone = var.zone
    }
  }
  
  maintenance_policy {
    auto_upgrade = true
    auto_repair  = true
    
    maintenance_window {
      day        = "monday"
      start_time = "23:00"
      duration   = "3h"
    }
  }
}

# Security Groups
resource "yandex_vpc_security_group" "k8s_master" {
  name        = "${var.cluster_name}-master-sg"
  description = "Security group for Kubernetes master"
  network_id  = var.network_id
  
  ingress {
    protocol       = "TCP"
    description    = "Kubernetes API"
    port           = 6443
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    protocol       = "TCP"
    description    = "SSH"
    port           = 22
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    protocol       = "ANY"
    description    = "Outgoing traffic"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}