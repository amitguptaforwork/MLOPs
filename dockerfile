FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD streamlit run step3.1_streamlitUI_UsingModelDirectly.py --server.port $PORT --server.address 0.0.0.0
