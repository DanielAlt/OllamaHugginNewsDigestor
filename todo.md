## Bugs and Features (Todo)

- [Bug] Some articles won't load via the 'requests' module, sites using ReactJS or 
 other such frameworks load all content from asynchronous calls after the initial
 page load. For such sites we need to 'detect' and load them in a headless 
 browser. see: https://github.com/browserless/browserless 

- -[Feat] Move prompts and `.env` to the `%APP_DATA%` installation directory
 in order to make the system prompts editable for the user and per the 
 program instance.-
 - [Feat] Once the above Feat is complete, create a configuration wizard tool 
  for easier onboarding of new users. 

- [Bug] [Audio Generation] We are voice Cloning on each run: cloning the voice 
 once, and storing the voice in the installation directory would be more optimal


- [Feat] Add a CVE lookup tool to enrich Vulnerability Information ** \
 consider 
 https://docs.opencve.io/
 https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=CVE-2025-27152


- [Feat] Article categorization
- [Feat] Article 'Business Vertical' (try to determine the business vertical 
 affected by the Article. Certain threats are industry specific, and people 
 are interested in their industry vertical specifically.  

