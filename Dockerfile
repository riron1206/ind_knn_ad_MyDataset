FROM sinpcw/pytorch:2.0.0

RUN pip install ipyplot && \
    pip install wget && \
    pip install torchinfo && \
    pip install thop && \
    pip install papermill && \
    pip uninstall -y ipywidgets && \
    pip install ipywidgets && \
    pip install -U timm
