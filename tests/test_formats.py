import pytest

from apud_bench.formats import FormatError, load

PAGE_2019 = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageFilename="f001r.jpg" imageWidth="1000" imageHeight="2000">
    <TextRegion id="r1">
      <Coords points="0,0 1000,0 1000,2000 0,2000"/>
      <TextLine id="l1">
        <Coords points="100,200 600,200 600,260 100,260"/>
        <Baseline points="100,250 600,250"/>
        <TextEquiv index="1"><Unicode>variant reading</Unicode></TextEquiv>
        <TextEquiv index="0"><Unicode>In principio erat</Unicode></TextEquiv>
      </TextLine>
      <TextLine id="l2">
        <Coords points="100,300 600,300 600,360 100,360"/>
        <Word id="w1"><Coords points="100,300 200,300 200,360"/>
          <TextEquiv><Unicode>verbum</Unicode></TextEquiv></Word>
        <Word id="w2"><Coords points="220,300 300,300 300,360"/>
          <TextEquiv><Unicode>Dei</Unicode></TextEquiv></Word>
      </TextLine>
    </TextRegion>
    <TextLine id="orphan"><Coords points="10,10 20,10 20,20"/></TextLine>
  </Page>
</PcGts>"""

PAGE_2010 = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19">
  <Page imageFilename="b.jpg" imageWidth="400" imageHeight="500">
    <TextRegion id="r1">
      <TextLine id="l1">
        <Coords><Point x="10" y="20"/><Point x="110" y="20"/><Point x="110" y="60"/></Coords>
        <TextEquiv><Unicode>Bentham</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>"""

ALTO_ESCRIPTORIUM = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Description>
    <MeasurementUnit>pixel</MeasurementUnit>
    <sourceImageInformation><fileName>btv1b_f12.jpg</fileName></sourceImageInformation>
  </Description>
  <Layout><Page WIDTH="2000.0" HEIGHT="3000.0" PHYSICAL_IMG_NR="1" ID="p1"><PrintSpace>
    <TextBlock ID="b1">
      <TextLine ID="l1" BASELINE="100 180 900 185" HPOS="100" VPOS="120" WIDTH="800" HEIGHT="80">
        <Shape><Polygon POINTS="100 120 900 120 900 200 100 200"/></Shape>
        <String CONTENT="In" HPOS="100" VPOS="120" WIDTH="80" HEIGHT="80"/>
        <SP/>
        <String CONTENT="principio"/>
        <String CONTENT="erat"/>
        <HYP CONTENT="-"/>
      </TextLine>
      <TextLine ID="l2" BASELINE="100,280 900,285" HPOS="100" VPOS="220" WIDTH="800" HEIGHT="80">
        <Shape><Polygon POINTS="100,220 900,220 900,300 100,300"/></Shape>
        <String CONTENT="verbum"/>
      </TextLine>
    </TextBlock>
  </PrintSpace></Page></Layout>
</alto>"""

ALTO_RECT_ONLY = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Description><MeasurementUnit>pixel</MeasurementUnit></Description>
  <Layout><Page WIDTH="1000" HEIGHT="1000"><PrintSpace>
    <TextBlock ID="b1">
      <TextLine ID="abs" BASELINE="160" HPOS="10" VPOS="100" WIDTH="300" HEIGHT="60">
        <String CONTENT="absolute"/>
      </TextLine>
      <TextLine ID="rel" BASELINE="50" HPOS="10" VPOS="200" WIDTH="300" HEIGHT="60">
        <String CONTENT="relative"/>
      </TextLine>
    </TextBlock>
  </PrintSpace></Page></Layout>
</alto>"""

ALTO_MM10 = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Description><MeasurementUnit>mm10</MeasurementUnit></Description>
  <Layout><Page WIDTH="2000" HEIGHT="3000"><PrintSpace>
    <TextBlock ID="b1">
      <TextLine ID="l1" HPOS="200" VPOS="300" WIDTH="1000" HEIGHT="100">
        <String CONTENT="Transkribus"/>
      </TextLine>
    </TextBlock>
  </PrintSpace></Page></Layout>
</alto>"""


@pytest.fixture
def write(tmp_path):
    def _write(name, content):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    return _write


def test_page_2019(write):
    page = load(write("f001r.xml", PAGE_2019))
    assert (page.id, page.image, page.width, page.height) == ("f001r", "f001r.jpg", 1000, 2000)
    assert page.source_format == "page-2019-07-15"
    first, second, orphan = page.lines
    assert first.text == "In principio erat"  # lowest TextEquiv index wins
    assert first.bbox == (100, 200, 500, 60)
    assert first.baseline == ((100, 250), (600, 250))
    assert first.region_id == "r1"
    assert second.text == "verbum Dei"  # word-level fallback
    assert orphan.region_id is None


def test_page_rescaled_to_real_image(write):
    page = load(write("f001r.xml", PAGE_2019), image_size=(2000, 4000))
    assert (page.width, page.height) == (2000, 4000)
    assert page.lines[0].bbox == (200, 400, 1000, 120)


def test_page_2010_point_children(write):
    page = load(write("b.xml", PAGE_2010))
    assert page.source_format == "page-2010-03-19"
    assert page.lines[0].polygon == ((10, 20), (110, 20), (110, 60))
    assert page.lines[0].text == "Bentham"


def test_alto_escriptorium(write):
    page = load(write("f12.xml", ALTO_ESCRIPTORIUM))
    assert (page.image, page.width, page.height, page.source_format) == (
        "btv1b_f12.jpg",
        2000,
        3000,
        "alto-v4",
    )
    l1, l2 = page.lines
    assert l1.text == "In principio erat-"
    assert l1.polygon == ((100, 120), (900, 120), (900, 200), (100, 200))
    assert l1.baseline == ((100, 180), (900, 185))
    assert l2.polygon[0] == (100, 220) and l2.baseline == ((100, 280), (900, 285))
    assert l1.region_id == "b1"


def test_alto_rectangle_and_scalar_baseline(write):
    absolute, relative = load(write("r.xml", ALTO_RECT_ONLY)).lines
    assert absolute.bbox == (10, 100, 300, 60)
    assert absolute.baseline == ((10, 160), (310, 160))
    assert relative.baseline == ((10, 250), (310, 250))


def test_alto_mm10_needs_image_size(write):
    path = write("t.xml", ALTO_MM10)
    with pytest.raises(FormatError, match="image size is required"):
        load(path)
    page = load(path, image_size=(4000, 6000))
    assert page.lines[0].bbox == (400, 600, 2000, 200)


def test_unsupported_root(write):
    with pytest.raises(FormatError, match="unsupported root"):
        load(write("x.xml", "<TEI/>"))


def test_to_dict_is_json_ready(write):
    data = load(write("f001r.xml", PAGE_2019)).to_dict()
    assert data["lines"][0]["bbox"] == [100, 200, 500, 60]
    assert data["lines"][0]["polygon"][0] == [100.0, 200.0]


def test_cli_inspect(write, capsys):
    import json

    from apud_bench.cli import main

    assert main(["inspect", str(write("f001r.xml", PAGE_2019)), "--image-size", "2000x4000"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["width"] == 2000 and data["lines"][0]["bbox"] == [200, 400, 1000, 120]
