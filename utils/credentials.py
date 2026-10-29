"""Local, optional Hugging Face credentials; never print their contents."""
import json
import os
from pathlib import Path


def read_api_key(path='api_key.json') -> dict[str, str]:
    """Create an empty private file if absent, without replacing existing credentials."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, 'w') as stream:
            json.dump({'HUGGING_FACE_API_KEY': ''}, stream)
            stream.write('\n')
    try:
        data = json.loads(path.read_text())
    except (ValueError, UnicodeError):
        raise ValueError('Invalid API-key JSON; use api_key.example.json as the template.') from None
    if not isinstance(data, dict) or not isinstance(data.get('HUGGING_FACE_API_KEY', ''), str):
        raise ValueError('HUGGING_FACE_API_KEY must be a string (empty is allowed).')
    return {'HUGGING_FACE_API_KEY': data.get('HUGGING_FACE_API_KEY', '').strip()}


if __name__ == '__main__':
    read_api_key()
    print('api_key.json is ready; existing credentials were preserved.')
