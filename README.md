# rc-asistente-selena



## Getting started


## instalar en sistemas posix


```
sudo apt update
sudo apt install python3 python3-venv
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

```

## descargar punkt

```
python3

>>>> import nltk
>>>> nltk.download('punkt')

```


## Ejecutar el API

```
gunicorn -w 4 -b 0.0.0.0:5000 app:app

```


***
