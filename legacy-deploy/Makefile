create-inference-docker:
	docker build -f dockers/inference/Dockerfile -t wwvm .

serve-local:
	docker run -p 8080:8080 -e MODEL_VERSION=1.0.0 wwvm serve

ww-client-streamlit-run:
	streamlit run apps/streamlit-ww-client/streamlit_ww_client.py --server.port 4001