# Lambda execution role
resource "aws_iam_role" "lambda_exec" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

# Attach basic execution policy (CloudWatch logs)
resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Policy for EventBridge access (self-scheduling)
resource "aws_iam_role_policy" "eventbridge_access" {
  name = "${var.project_name}-eventbridge-access"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "events:PutRule",
          "events:PutTargets",
          "events:DisableRule",
          "events:EnableRule",
          "events:DescribeRule"
        ]
        Resource = "arn:aws:events:${var.aws_region}:*:rule/odds-*"
      }
    ]
  })
}

# Lambda function
resource "aws_lambda_function" "odds_scheduler" {
  filename         = "../lambda_deployment.zip"
  function_name    = var.project_name
  role            = aws_iam_role.lambda_exec.arn
  handler         = "lambda_handler.lambda_handler"
  runtime         = "python3.11"
  timeout         = var.lambda_timeout
  memory_size     = var.lambda_memory_size
  source_code_hash = filebase64sha256("../lambda_deployment.zip")

  environment {
    variables = {
      SCHEDULER_BACKEND = "aws"
      DATABASE_URL      = var.database_url
      ODDS_API_KEY      = var.odds_api_key
      LAMBDA_ARN        = "arn:aws:lambda:${var.aws_region}:${data.aws_caller_identity.current.account_id}:function:${var.project_name}"

      # Optional: Configure other settings
      LOG_LEVEL         = "INFO"
      ENABLE_VALIDATION = "true"

      # Note: AWS_REGION is automatically provided by Lambda runtime
      # No need to set it manually - it will be available in the environment
    }
  }

  # VPC configuration (if database requires it)
  # vpc_config {
  #   subnet_ids         = var.subnet_ids
  #   security_group_ids = var.security_group_ids
  # }
}

# CloudWatch log group for Lambda
# Note: Lambda automatically creates log groups, but we manage it explicitly for retention
resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${aws_lambda_function.odds_scheduler.function_name}"
  retention_in_days = 14

  # Prevent errors if log group already exists (e.g., created by Lambda or leftover from failed deploy)
  lifecycle {
    ignore_changes = []
  }
}

# Data source for current AWS account
data "aws_caller_identity" "current" {}

# Output Lambda ARN
output "lambda_function_arn" {
  description = "ARN of the Lambda function"
  value       = aws_lambda_function.odds_scheduler.arn
}

output "lambda_function_name" {
  description = "Name of the Lambda function"
  value       = aws_lambda_function.odds_scheduler.function_name
}
