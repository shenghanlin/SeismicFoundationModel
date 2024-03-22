# 🌟 Seismic pretrain

  Data link [<a href='https://rec.ustc.edu.cn/share/07e7c9a0-e83a-11ee-9663-ada87855acba' target='_blank'>DatFile]

The folder '''mae_data_more'' contains 2286422 224*224 seismic data. Limited by the size of the uploaded file, we split the zip file into eight sub-files. 
When decompressing on '''windows''', you only need to decompress the mae_data_moreb.zip to parse the other volumes together.

When decompressing on '''Linux''', you need to use the following command to synthesize a whole file before decompressing it.
‘’‘
zip -s 0 mae_data_more.zip --out pretrain.zip
unzip pretrain.zip
’‘’

All dat files are float32 binary files.

<br>
<div>
# License
This project is released under the [MIT license](LICENSE).

