# ECR Repository for Lambda container images
resource "aws_ecr_repository" "lambda" {
  name                 = "odds-scheduler"
  image_tag_mutability = "MUTABLE"

  # Enable image scanning for security
  image_scanning_configuration {
    scan_on_push = true
  }

  # Encryption at rest
  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name        = "odds-scheduler"
    Project     = "betting-odds-pipeline"
    Environment = "shared"
    ManagedBy   = "terraform"
  }
}

# Lifecycle policy - Conservative storage
# Dev: Keep only latest (1 image)
# Prod: Keep last 5 images
resource "aws_ecr_lifecycle_policy" "lambda" {
  repository = aws_ecr_repository.lambda.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep only latest dev image"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["dev-"]
          countType     = "imageCountMoreThan"
          countNumber   = 1
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Keep last 5 prod images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["prod-"]
          countType     = "imageCountMoreThan"
          countNumber   = 5
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 3
        description  = "Expire untagged images after 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# Outputs
output "ecr_repository_url" {
  description = "ECR repository URL for Lambda images"
  value       = aws_ecr_repository.lambda.repository_url
}

output "ecr_repository_arn" {
  description = "ECR repository ARN"
  value       = aws_ecr_repository.lambda.arn
}
