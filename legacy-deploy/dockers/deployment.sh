MODEL_IMAGE=$1
MODEL_VERSION=$2

ROLE_ARN="arn:aws:iam::676153624229:role/comcast-cross-account-role"
SESSION_NAME="comcast_cross_account_role"

ROLE_CREDS=$(aws sts assume-role --role-arn $ROLE_ARN --role-session-name $SESSION_NAME)
echo $ROLE_CREDS

export AWS_ACCESS_KEY_ID=$(echo $ROLE_CREDS | jq -r '.Credentials.AccessKeyId')
export AWS_SECRET_ACCESS_KEY=$(echo $ROLE_CREDS | jq -r '.Credentials.SecretAccessKey')
export AWS_SESSION_TOKEN=$(echo $ROLE_CREDS | jq -r '.Credentials.SessionToken')

aws sts get-caller-identity

ROLE_ARN="arn:aws:iam::585989164753:role/OneCloud/vrex-sky-role"
SESSION_NAME="vrex_sky_role"
ROLE_CREDS=$(aws sts assume-role --role-arn $ROLE_ARN --role-session-name $SESSION_NAME)
echo $ROLE_CREDS

export AWS_ACCESS_KEY_ID=$(echo $ROLE_CREDS | jq -r '.Credentials.AccessKeyId')
export AWS_SECRET_ACCESS_KEY=$(echo $ROLE_CREDS | jq -r '.Credentials.SecretAccessKey')
export AWS_SESSION_TOKEN=$(echo $ROLE_CREDS | jq -r '.Credentials.SessionToken')

aws sts get-caller-identity

# Perform Docker Login to obtain Credentials for push to Sky-WuW ECR within Comcast AWS Account.
aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin 585989164753.dkr.ecr.eu-west-2.amazonaws.com
# Tag Model Image for Push.
docker tag $MODEL_IMAGE 585989164753.dkr.ecr.eu-west-2.amazonaws.com/sky-wuw:latest
docker tag $MODEL_IMAGE 585989164753.dkr.ecr.eu-west-2.amazonaws.com/sky-wuw:$MODEL_VERSION
# Push Image to Comcast ECR Repo.
docker push 585989164753.dkr.ecr.eu-west-2.amazonaws.com/sky-wuw:$MODEL_VERSION