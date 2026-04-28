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

# Policy for S3 model bucket read access
resource "aws_iam_role_policy" "s3_model_read" {
  name = "${var.project_name}-s3-model-read"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:HeadObject"
        ]
        Resource = "arn:aws:s3:::${var.model_bucket_name}/*"
      }
    ]
  })
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
        Resource = "arn:aws:events:${var.aws_region}:*:rule/${var.rule_prefix}-*"
      }
    ]
  })
}

# Policy for SSM access (API key rotation state)
resource "aws_iam_role_policy" "ssm_api_key" {
  name = "${var.project_name}-ssm-api-key"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:PutParameter"
        ]
        Resource = "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/${var.project_name}/active-api-key-index"
      }
    ]
  })
}

# SSM parameter for active API key index (Lambda manages the value at runtime)
resource "aws_ssm_parameter" "active_api_key_index" {
  name  = "/${var.project_name}/active-api-key-index"
  type  = "String"
  value = "0"

  lifecycle {
    ignore_changes = [value]
  }
}

# Read-only access to the Betfair cert/key SSM SecureString parameters.
# The parameter values themselves are populated out-of-band by an operator
# (`aws ssm put-parameter --type SecureString ...`) so cert material is never
# stored in terraform state, GH Actions secrets, or the Lambda image.
resource "aws_iam_role_policy" "ssm_betfair_cert" {
  count = var.betfair_enabled ? 1 : 0

  name = "${var.project_name}-ssm-betfair-cert"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["ssm:GetParameter"]
        Resource = [
          "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/${var.project_name}/betfair/cert_pem",
          "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/${var.project_name}/betfair/key_pem",
        ]
      }
    ]
  })
}

# Lambda function (container image deployment)
resource "aws_lambda_function" "odds_scheduler" {
  function_name = var.project_name
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/odds-scheduler:${var.image_tag}"
  timeout       = var.lambda_timeout
  memory_size   = var.lambda_memory_size

  environment {
    variables = {
      SCHEDULER_BACKEND = "aws"
      DATABASE_URL      = var.database_url
      ODDS_API_KEY      = var.odds_api_key
      ODDS_API_KEYS     = var.odds_api_keys
      SSM_API_KEY_INDEX = "/${var.project_name}/active-api-key-index"
      LAMBDA_ARN        = "arn:aws:lambda:${var.aws_region}:${data.aws_caller_identity.current.account_id}:function:${var.project_name}"
      RULE_PREFIX       = var.rule_prefix

      # Optional: Configure other settings
      LOG_LEVEL           = "INFO"
      ENABLE_VALIDATION   = "true"
      MODEL_BUCKET        = var.model_bucket_name
      MODEL_NAME          = var.model_name
      DISCORD_WEBHOOK_URL = var.discord_webhook_url
      ALERT_ENABLED       = var.discord_webhook_url != "" ? "true" : "false"

      # Betfair Exchange direct ingestion. Cert PEM contents live in SSM
      # SecureString and are materialized to /tmp at cold start so we don't
      # bake cert material into the image or burn the 4 KB Lambda env-var
      # ceiling on PEMs.
      BETFAIR_ENABLED            = var.betfair_enabled ? "true" : "false"
      BETFAIR_USERNAME           = var.betfair_username
      BETFAIR_PASSWORD           = var.betfair_password
      BETFAIR_APP_KEY            = var.betfair_app_key
      BETFAIR_CERT_PEM_SSM_PARAM = "/${var.project_name}/betfair/cert_pem"
      BETFAIR_KEY_PEM_SSM_PARAM  = "/${var.project_name}/betfair/key_pem"
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
