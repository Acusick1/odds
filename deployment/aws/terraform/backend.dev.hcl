bucket         = "odds-scheduler-terraform-state"
key            = "dev/terraform.tfstate"
region         = "eu-west-1"
encrypt        = true
dynamodb_table = "odds-scheduler-terraform-locks"
