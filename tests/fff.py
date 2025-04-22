
import os
import ssl
import json
import os.path as osp
import urllib.request


def download_file(file_path):
    fp = os.path.join(file_path, 'alpaca_data.json')
    download_url(
        'https://raw.githubusercontent.com/tatsu-lab'
        '/stanford_alpaca/'
        '761dc5bfbdeeffa89b8bff5d038781a4055f796a/'
        'alpaca_data.json', file_path)
    list_data_dict = load_json(fp)
    return list_data_dict


def download_url(url: str, folder='folder'):
    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path


def load_json(file_path,
              instruction='instruction',
              input='input',
              output='output',
              category='category'):
    # Format: [{'instruction': ..., 'input': ..., 'output':...}]
    with open(file_path, 'r', encoding="utf-8") as f:
        list_data_dict = json.load(f)

    # Replace key
    new_list_data_dict = []
    for item in list_data_dict:
        new_item = dict(
            instruction=item[instruction] if instruction in item else None,
            input=item[input] if input in item else None,
            output=item[output] if output in item else None,
            category=item[category] if category in item else None)
        new_list_data_dict.append(new_item)
    return new_list_data_dict


if __name__ == '__main__':
    _file_path = '/PycharmProjects/federatedscope/data'
    download_file(_file_path)
