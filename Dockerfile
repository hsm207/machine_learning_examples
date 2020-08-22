FROM continuumio/anaconda3:2020.07

RUN conda install -c conda-forge altair vega_datasets
RUN conda install black

EXPOSE 8888

CMD jupyter notebook --allow-root --ip 0.0.0.0