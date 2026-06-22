## Bugs and Features (Todo)

- [Bug] Some articles won't load via the 'requests' module, sites using ReactJS or 
 other such frameworks load all content from asynchronous calls after the initial
 page load. For such sites we need to 'detect' and load them in a headless 
 browser. see: https://github.com/browserless/browserless 

- [Feat] Move prompts and `.env` to the `%APP_DATA%` installation directory
 in order to make the system prompts editable for the user and per the 
 program instance. 
 - [Feat] Once the above Feat is complete, create a configuration wizard tool 
  for easier onboarding of new users. 

- [Bug] [Audio Generation] We are voice Cloning on each run: cloning the voice 
 once, and storing the voice in the installation directory would be more optimal

- [Feat] Add a CVE lookup tool to enrich Vulnerability Information

- [Feat] Add a Main database. Maintain a SQlite3 database of Article URLs,
 titles, and descriptions, and published_at dates. As well as IOC values. 
 Use it to maintain a 'memory' of events; in order to not report duplicate 
 intelligence, or duplicate articles in a given timeframe. 

- [Feat] Article categorization
- [Feat] Article 'Business Vertical' (try to determine the business vertical 
 affected by the Article. Certain threats are industry specific, and people 
 are interested in their industry vertical specifically.  
