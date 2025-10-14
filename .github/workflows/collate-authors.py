import jupytext
import glob
import os
import numpy as np
import yaml

EXCLUDED = ['README.md', 'index.md', 'template.md']


if __name__ == '__main__':

    md_files = glob.glob(os.path.join(os.environ['GITHUB_WORKSPACE'], '**/*.md'), recursive=True)

    filt_md_files = [md_f for md_f in md_files if all([ex_patt not in md_f for ex_patt in EXCLUDED])]

    comb_auths = {}
    for cur_md in filt_md_files:

        with open(cur_md, 'r') as md_read:
            md_jupy = jupytext.reads(md_read.read())

        if 'authors' not in md_jupy.metadata:
            continue

        cur_auth_info = md_jupy.metadata['authors']
        for auth_ent in cur_auth_info:
            auth_name = auth_ent['name']
            comb_auths.setdefault(auth_name, {'affiliations': []})

            try:
                rel_affs = [] if 'affiliations' not in auth_ent else [auth_ent['affiliations']] if isinstance(
                    auth_ent['affiliations'], str) else auth_ent['affiliations']
                comb_auths[auth_name]['affiliations'] += rel_affs
            except TypeError:
                pass

            if 'orcid' in auth_ent and isinstance(auth_ent['orcid'], str):
                orc_id = auth_ent['orcid'].split('/')[-1]
                if len(orc_id) != 19:
                    continue
                else:
                    comb_auths[auth_name]['orcid'] = orc_id

            if 'github' in auth_ent and isinstance(auth_ent['github'], str):
                comb_auths[auth_name]['github'] = "https://github.com/" + auth_ent['github'].split('/')[-1]

            if 'email' in auth_ent and isinstance(auth_ent['email'], str):
                comb_auths[auth_name]['email'] = auth_ent['email']

            if 'website' in auth_ent and isinstance(auth_ent['website'], str):
                comb_auths[auth_name]['website'] = auth_ent['website']

    for auth, all_info in comb_auths.items():
        aff = np.array(list(set(all_info['affiliations'])))
        if len(aff) > 1:
            nasa_where = np.argwhere(
                (np.char.find(aff, 'NASA Goddard') != -1) | (np.char.find(aff, 'HEASARC') != -1)).flatten()

            pop_elem = aff[nasa_where]
            aff = np.concatenate([np.delete(aff, nasa_where), pop_elem])

        comb_auths[auth]['affiliations'] = aff.tolist()

    comb_auths = {auth_key: comb_auths[auth_key] for auth_key in sorted(comb_auths)}

    with open(os.path.join(os.environ['GITHUB_WORKSPACE'], 'CONTRIBUTORS.yml'), 'w') as contrib_write:
        contrib_write.write(yaml.dump(comb_auths))


    # -------------------------- BUILDING THE AUTHORS PAGE --------------------------

    html_block = """
        <table class="author-table">
            <thead>
                <tr>
                    <th class="author-header">Author</th>
                    <th class="affiliation-header">Affiliation</th>
                </tr>
            </thead>
            <tbody>
                {auth_block}
            </tbody>
        </table>
        """

    author_temp = """
        <tr>
            <td rowspan="{num_row}" class="author-block author-separator">
                <div class="author-info-block">
                    {author_name_block}
                    {orcid_block}
                </div>
                {email_block}
            </td>
            <td class="affiliation-item no-spacing {author_separator}">
                {aff}
            </td>
        </tr>
        """

    name_temp = """
        <span class="author-name-text">{name}</span>
        """

    name_link_temp = """
        <span class="author-name-text"><a href={website}>{name}</a></span>
        """

    email_temp = """
        <div class="author-email"><a href="mailto:{email}">{email}</a></div>
        """

    orcid_temp = """
        <sup class="orcid-sup">
            <a href="https://orcid.org/{orc_id}" target="_blank" title="View ORCiD record">
                <img src="_static/ORCID-iD_icon_vector.svg" alt="ORCID Logo" class="orcid-icon"/>
            </a>
        </sup>
        """

    extra_aff_temp = """
        <tr>
            <td class="affiliation-item no-spacing {author_separator}">
                {aff}
            </td>
        </tr>
        """

    all_auth_block = ""

    for auth, all_info in comb_auths.items():
        num_aff = len(all_info['affiliations'])

        if 'website' not in all_info:
            cur_name_block = name_temp.format(name=auth)
        else:
            cur_name_block = name_link_temp.format(name=auth, website=all_info['website'])

        if 'orcid' in all_info:
            cur_orcid_block = orcid_temp.format(orc_id=all_info['orcid'])
        else:
            cur_orcid_block = ''

        if 'email' in all_info:
            cur_email_block = email_temp.format(email=all_info['email'])
            cur_row_span = max(2, num_aff)
        else:
            cur_email_block = ''
            cur_row_span = num_aff

        if num_aff > 1:
            auth_block_add_class = ''
        else:
            auth_block_add_class = 'author-separator'

        if num_aff == 0:
            prim_aff = ""
        else:
            prim_aff = all_info['affiliations'][0]

        cur_auth_block = author_temp.format(num_row=cur_row_span, author_name_block=cur_name_block,
                                            orcid_block=cur_orcid_block, email_block=cur_email_block,
                                            author_separator=auth_block_add_class, aff=prim_aff)

        # Extra affiliation bits
        for cur_aff_ind, cur_aff in enumerate(all_info['affiliations'][1:]):
            extra_add_class = "" if cur_aff_ind == (len(all_info['affiliations'][1:]) + 1) else 'author-separator'
            cur_auth_block += extra_aff_temp.format(author_separator=extra_add_class, aff=cur_aff)

        all_auth_block += cur_auth_block

    final_html = html_block.format(auth_block=all_auth_block)


    with open(os.path.join(os.environ['GITHUB_WORKSPACE'], 'authors.md'), 'w') as author_page:
        author_page.writelines(['# Contributors\n\n', '```{raw} html'])
        author_page.write(final_html)
        author_page.write('\n```')
