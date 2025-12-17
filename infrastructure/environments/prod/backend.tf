terraform {
  backend "s3" {
    bucket                      = "terraform-state-ml-prod"
    key                         = "terraform.tfstate"
    region                      = "ru-7"
    endpoint                    = "https://api.storage.selcloud.ru"
    access_key                  = "YOUR_ACCESS_KEY"
    secret_key                  = "YOUR_SECRET_KEY"
    skip_region_validation      = true
    skip_credentials_validation = true
    force_path_style            = true
  }
}