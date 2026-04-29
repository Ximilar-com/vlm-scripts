
export MODEL_URL=""
export FOLDER_PATH="tmp/models/conversion_936c3bb7-e055-4dd8-a0a3-6fc2bcfe036d"
wget \
  --output-document "$FOLDER_PATH".zip
  "$MODEL_URL"
unzip -o "$FOLDER_PATH".zip -d $FOLDER_PATH
ls -la "$FOLDER_PATH"
rm "$FOLDER_PATH".zip
