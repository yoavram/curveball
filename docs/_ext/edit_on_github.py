"""
Sphinx extension to add ReadTheDocs-style "Edit on GitHub" links to the
sidebar.

Loosely based on https://github.com/astropy/astropy/pull/347

Curveball note: downloaded from https://gist.github.com/MantasVaitkunas/7c16de233812adcb7028#file-edit_on_github-py-L38-L43
and slightly modified to work with Curveball.
"""

import os
import warnings


__licence__ = 'BSD (3 clause)'


def get_github_url(app, view, path):
    return 'https://github.com/{project}/{view}/{branch}/docs/{path}'.format(
        project=app.config.edit_on_github_project,
        view=view,
        branch=app.config.edit_on_github_branch,
        path=path)    


def html_page_context(app, pagename, templatename, context, doctree):
    if templatename != 'page.html':
        return

    if not app.config.edit_on_github_project:
        warnings.warn("edit_on_github_project not specified")
        return
    
    if doctree is None:
        #warnings.warn("doctree is None for page {0}".format(pagename))
        return

    path = os.path.relpath(doctree.get('source'), app.builder.srcdir)
    show_url = get_github_url(app, 'blob', path)
    edit_url = get_github_url(app, 'edit', path)

    context['show_on_github_url'] = show_url
    context['edit_on_github_url'] = edit_url

    # For sphinx_rtd_theme.
    context['display_github'] = True
    context['github_user'] = app.config.edit_on_github_project.split('/')[0]
    context['github_version'] = app.config.edit_on_github_branch + '/docs/'
    context['github_repo'] = app.config.edit_on_github_project.split('/')[1]
    context['source_suffix'] = app.config.source_suffix[0]


def setup(app):
    app.add_config_value('edit_on_github_project', '', True)
    app.add_config_value('edit_on_github_branch', 'master', True)
    app.connect('html-page-context', html_page_context)
