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
                    comb_auths[auth_name]['orcid'] = "https://orcid.org/" + orc_id

            if 'github' in auth_ent and isinstance(auth_ent['github'], str):
                comb_auths[auth_name]['github'] = "https://github.com/" + auth_ent['github'].split('/')[-1]

            if 'email' in auth_ent and isinstance(auth_ent['email'], str):
                comb_auths[auth_name]['email'] = auth_ent['email']

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
