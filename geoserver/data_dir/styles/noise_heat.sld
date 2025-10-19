<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor xmlns="http://www.opengis.net/sld"
    xmlns:se="http://www.opengis.net/se"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    xmlns:ogc="http://www.opengis.net/ogc"
    version="1.1.0">
  <NamedLayer>
    <se:Name>noise_heat</se:Name>
    <UserStyle>
      <se:Name>noise_heat</se:Name>
      <se:FeatureTypeStyle>
        <se:Rule>
          <se:RasterSymbolizer>
            <se:Opacity>0.7</se:Opacity>
            <se:ChannelSelection>
              <se:GrayChannel>
                <se:SourceChannelName>1</se:SourceChannelName>
              </se:GrayChannel>
            </se:ChannelSelection>
            <se:ColorMap type="ramp" extended="true">
              <se:ColorMapEntry color="#00FF00" quantity="35" opacity="0.2"/>
              <se:ColorMapEntry color="#ADFF2F" quantity="45" opacity="0.4"/>
              <se:ColorMapEntry color="#FFFF00" quantity="55" opacity="0.55"/>
              <se:ColorMapEntry color="#FFA500" quantity="65" opacity="0.6"/>
              <se:ColorMapEntry color="#FF4500" quantity="75" opacity="0.65"/>
              <se:ColorMapEntry color="#FF0000" quantity="85" opacity="0.7"/>
              <se:ColorMapEntry color="#8B0000" quantity="95" opacity="0.75"/>
            </se:ColorMap>
          </se:RasterSymbolizer>
        </se:Rule>
      </se:FeatureTypeStyle>
    </UserStyle>
  </NamedLayer>
</StyledLayerDescriptor>
