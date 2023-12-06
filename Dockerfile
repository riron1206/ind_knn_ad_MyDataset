FROM sinpcw/pytorch:2.0.0

RUN pip install ipyplot && \
    pip install iterative-stratification && \
    pip install wget && \
    pip install torchinfo && \
    pip install thop && \
    pip install papermill && \
    pip install jupyterlab && \
    pip install torchmetrics && \
    pip install seaborn && \
    pip install streamlit && \
    pip install gradio && \
    pip uninstall -y ipywidgets && \
    pip install ipywidgets && \
    pip install -U timm

# for gradio
# https://qiita.com/db0091/items/cf03c59bb4138d9a4dbb
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860
