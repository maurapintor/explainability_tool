class XMLEntityResolver {
    public InputSource resolveEntity(String publicId, String systemId) throws SAXException, IOException {
        if (systemId != null) {
            if (LOGGER.isLoggable(Level.FINE)) {
                LOGGER.fine("Will try to resolve systemId [" + systemId + "]");
            }
            if (systemId.startsWith(TESTNG_NAMESPACE)) {
                LOGGER.fine("It's a TestNG document, will try to lookup DTD in classpath");
                String dtdFileName = systemId.substring(TESTNG_NAMESPACE.length());
                URL url = getClass().getClassLoader().getResource(dtdFileName);
                if (url != null)
                    return new InputSource(url.toString());
            }
        }
        return new InputSource();
    }
}