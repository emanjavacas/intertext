<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:xs="http://www.w3.org/2001/XMLSchema"
    exclude-result-prefixes="xs" xmlns:tei="http://www.tei-c.org/ns/1.0"
    version="1.0">
    <xsl:output method="text" encoding="UTF-8"/>
    <xsl:template match="tei:note">
        <xsl:text> </xsl:text>
    </xsl:template>
    <xsl:template match="tei:p|tei:div|tei:seg|tei:l|tei:w|tei:ab">
        <xsl:text> </xsl:text><xsl:apply-templates />
    </xsl:template>
    <xsl:template match="tei:label|tei:ref|tei:milestone|tei:orig|tei:abbr|tei:head|tei:title|tei:teiHeader|tei:del|tei:g|tei:bibl|tei:front|tei:back|tei:foreign" />
</xsl:stylesheet>