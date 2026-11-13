"""Check generated local HTML links/assets/anchors using only Python stdlib."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / 'site'
PREFIX = '/BoseHubbardNQS/'


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if 'id' in data:
            self.ids.add(data['id'])
        key = 'href' if tag in ('a', 'link') else 'src' if tag in ('img', 'script') else None
        if key and data.get(key):
            self.links.append(data[key])


def main():
    pages = {path: Page(path.read_text()) for path in SITE.rglob('*.html')}
    if not pages:
        raise SystemExit('No generated HTML found; run zensical build --strict first')
    checked = 0
    failures = []
    for path, page in pages.items():
        for value in page.links:
            url = urlsplit(value)
            if url.scheme or url.netloc:
                continue
            route = unquote(url.path)
            if route.startswith(PREFIX):
                destination = SITE / route[len(PREFIX):]
            elif route.startswith('/'):
                destination = SITE / route.lstrip('/')
            else:
                destination = path.parent / route if route else path
            destination = destination.resolve()
            if destination.is_dir():
                destination /= 'index.html'
            if not destination.is_file():
                failures.append(f'{path.relative_to(SITE)} -> {value}: missing file')
            elif url.fragment and destination.suffix == '.html':
                target = pages.get(destination) or Page(destination.read_text())
                if unquote(url.fragment) not in target.ids:
                    failures.append(f'{path.relative_to(SITE)} -> {value}: missing anchor')
            checked += 1
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'{len(pages)} HTML pages and {checked} local links/assets/anchors verified')


if __name__ == '__main__':
    main()
