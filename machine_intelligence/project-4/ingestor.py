#!/bin/python3
#This code is designed to bring in a subset of data from online sources and create generated features from it.

import requests, whois, json

#domains
benign_datasets = []

#domain lists
#malicious_datasource = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links/output/domains/ACTIVE/list"
malicious_datasource = 'https://raw.githubusercontent.com/HexxiumCreations/threat-list/gh-pages/domainsonly'
benign_datasource = "https://raw.githubusercontent.com/Kikobeats/top-sites/master/top-sites.json"

#output filesource
output = "./ingested_data.json"

def get_data():
    benign_raw = requests.get(benign_datasource)
    malicious_raw = requests.get(malicious_datasource)
    benign_raw.connection.close()
    malicious_raw.connection.close()
    for i in json.loads(benign_raw.content):
        yield([0, i['rootDomain']])
    for i in str(malicious_raw.content).split("\\n"):
        yield([1, i])
        
def get_features(domain, hamspam):
    length = len(domain)
    numcount, letters, symbols = 0,0,0
    #split by tld, grab beginning
    for i in domain.split('.')[0]:
        if(i.isdigit() == True):
            numcount = numcount+1
        elif(i.isalpha()):
            letters = letters+1
        else:
            symbols = symbols+1
    #grab whois data
    try:
        whois_data = whois.whois(domain)
        whois_expire_date = str(whois_data['expiration_date'])
        whois_create_date = str(whois_data['creation_date'])
        registrar = whois_data['registrar']
    except: 
        whois_data = None
        whois_create_date = None
        whois_expire_date = None
        registrar = None
    #export as json
    return json.dumps([hamspam, domain, length, numcount, letters, symbols, whois_expire_date, whois_create_date, registrar])
            
f = open("ingested_json", "w")
for i in get_data():
    f.write(get_features(i[1], i[0]))
f.close()
    