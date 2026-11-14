"""Third-party-derived helpers, kept as standalone scripts so they can be run or audited alone.

Each is executed by path (`runpy`) rather than imported, because each is also a usable
command-line tool and because keeping them out of the import graph makes it obvious that the
pipeline's results do not depend on anything here being importable. See README.md.
"""
