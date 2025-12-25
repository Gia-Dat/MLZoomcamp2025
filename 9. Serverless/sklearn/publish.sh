
ECR_URL=219826711426.dkr.ecr.us-east-1.amazonaws.com
REPO_URL=${ECR_URL}/churn-prediction-lambda

LOCAL_IMAGE=churn-prediction-lambda


aws ecr get-login-password \
  --region "us-east-1" \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}


REMOTE_IMAGE_TAG="${REPO_URL}:v1"

docker build --platform linux/amd64 -t ${LOCAL_IMAGE} .
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}

echo "done"