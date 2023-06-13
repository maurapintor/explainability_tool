class SuiteResult {
    static List<SuiteResult> parse(File xmlReport, boolean keepLongStdio, PipelineTestDetails pipelineTestDetails)
            throws DocumentException, IOException, InterruptedException {
        List<SuiteResult> r = new ArrayList<SuiteResult>();
        SAXReader saxReader = new SAXReader();
        setFeatureQuietly(saxReader, "http://xml.org/sax/features/external-general-entities", false);
        saxReader.setEntityResolver(new XMLEntityResolver());
        FileInputStream xmlReportStream = new FileInputStream(xmlReport);
        try {
            Document result = saxReader.read(xmlReportStream);
            Element root = result.getRootElement();
            parseSuite(xmlReport, keepLongStdio, r, root, pipelineTestDetails);
        } finally {
            xmlReportStream.close();
        }
        return r;
    }
}