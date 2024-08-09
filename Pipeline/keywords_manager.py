class KeywordsManager:
    def __init__(self, keywords=None):
        """
        Initialize the KeywordsManager with a list of keywords.
        
        Args:
            keywords (list): A list of keywords. If None, uses default keywords.
        """
        if keywords is None:
            self.keywords = [
                "nuclear safety", "nuclear security", "nuclear regulations", "nuclear industry",
                "nuclear act", "Canada Energy Regulator", "nuclear facility", "nuclear facilities",
                "CNSC", "Nuclear Safety and Control Act", "Canadian Nuclear Safety Commission",
                "CNSC regulatory documents", "Nuclear Facilities Regulations",
                "International Atomic Energy Agency", "IAEA Regulations", "IAEA", "IAEA Safety Glossary",
                "certification of prescribed nuclear equipment", "REGDOC", "RegDoc",
                "nuclear safety standards", "nuclear reactor safety", "radiation protection",
                "nuclear safety culture", "nuclear safety regulations", "nuclear plant safety",
                "nuclear safety analysis", "emergency preparedness nuclear", "nuclear safety protocols",
                "nuclear accident prevention", "safety of nuclear facilities", "nuclear safety management",
                "nuclear risk assessment", "nuclear safety engineering", "nuclear safety guidelines",
                "nuclear regulatory framework", "nuclear regulations compliance", "nuclear safety laws",
                "nuclear regulatory authority", "nuclear industry regulations", "nuclear regulatory standards",
                "nuclear licensing regulations", "nuclear regulatory policies", "nuclear security regulations",
                "nuclear regulatory compliance", "regulatory oversight nuclear", "nuclear energy regulation",
                "nuclear material regulations", "nuclear environmental regulations", "nuclear waste regulations",
                "nuclear security standards", "nuclear facility security", "nuclear security measures",
                "nuclear material security", "nuclear security regulations", "nuclear security protocols",
                "nuclear security threats", "nuclear security compliance", "nuclear security policies",
                "nuclear security frameworks", "nuclear security technology", "nuclear security law",
                "nuclear security incidents", "nuclear security assessments", "nuclear security strategy",
                "security of nuclear substances", "nuclear fission", "nuclear fusion", "radioactive decay",
                "half-life", "critical mass", "nuclear chain reaction", "neutron moderation", "nuclear reactor",
                "control rods", "nuclear fuel cycle", "radioactive waste management", "nuclear radiation",
                "alpha particles", "beta particles", "gamma rays", "neutron flux", "nuclear isotopes",
                "radioactive contamination", "nuclear meltdown", "radiation shielding", "nuclear power plant",
                "uranium enrichment", "plutonium reprocessing", "nuclear decommissioning", "nuclear proliferation",
                "nuclear safeguards", "radiation dosimetry", "thermal neutron", "fast neutron", "breeder reactor",
                "Atomic Energy of Canada", "nuclear material", "radiation protection", "code of practice",
                "REGDOC-3.6", "Atomic Energy of Canada Limited", "authorized nuclear operator",
                "boiling water reactor", "Canada Deuterium Uranium", "criticality accident sequence assessment",
                "Canadian Council of Ministers of the Environment", "Canadian Environmental Assessment Act",
                "certified exposure device operator", "Canadian Environmental Protection Act", "counterfeit",
                "curie", "Canadian Nuclear Safety Commission", "criticality safety control",
                "emergency core cooling system", "extended loss of AC power", "Federal Nuclear Emergency Plan",
                "fitness for duty", "fuel incident notification and analysis system", "gigabecquerel", "gray",
                "high-enriched uranium", "hydrogenated tritium oxide", "International Atomic Energy Agency",
                "irradiated fuel bay", "Institute of Nuclear Power Operations", "International Physical Protection Advisory Service",
                "International Reporting System for Operating Experience", "International Nuclear and Radiological Event Scale",
                "International Commission on Radiological Protection", "International Commission on Radiation Units and Measurements",
                "low-enriched uranium", "loss-of-coolant accident", "megabecquerel", "micro modular reactor",
                "nuclear criticality safety", "National Non-Destructive Testing Certification Body", "nuclear emergency management",
                "Nuclear Emergency Organization", "nuclear energy worker", "Nuclear Suppliers Group", "spent nuclear fuel",
                "safe operating envelope", "sievert", "International System of Units", "systems important to safety",
                "site selection threat", "risk assessment"
            ]
        else:
            self.keywords = keywords

    def get_keywords(self):
        """
        Get the list of keywords.
        
        Returns:
            list: A list of keywords.
        """
        return self.keywords

    def add_keywords(self, new_keywords):
        """
        Add new keywords to the existing list.
        
        Args:
            new_keywords (list): A list of new keywords to add.
        """
        if isinstance(new_keywords, list):
            self.keywords.extend(new_keywords)
        else:
            raise TypeError("New keywords must be provided as a list.")

    def remove_keywords(self, keywords_to_remove):
        """
        Remove specific keywords from the list.
        
        Args:
            keywords_to_remove (list): A list of keywords to remove.
        """
        if isinstance(keywords_to_remove, list):
            self.keywords = [keyword for keyword in self.keywords if keyword not in keywords_to_remove]
        else:
            raise TypeError("Keywords to remove must be provided as a list.")

    def update_keywords(self, keywords):
        """
        Update the entire list of keywords.
        
        Args:
            keywords (list): A new list of keywords to replace the old list.
        """
        if isinstance(keywords, list):
            self.keywords = keywords
        else:
            raise TypeError("Keywords must be provided as a list.")
