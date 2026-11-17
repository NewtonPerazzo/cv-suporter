FROM ollama/ollama:0.34.4
COPY docker/ollama/start.sh /start.sh
RUN chmod +x /start.sh
ENTRYPOINT ["/bin/sh", "/start.sh"]
