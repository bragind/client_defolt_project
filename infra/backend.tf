# infra/backend.tf
terraform {
  backend "s3" {
    endpoint   = "https://storage.yandexcloud.net"
    bucket     = "my-credit-scoring-state-bucket"
    region     = "ru-central1"
    key        = "terraform.tfstate"
    skip_requesting_account_id = true
    skip_credentials_validation = true
    skip_metadata_api_check = true
    force_path_style = true
  }
}