/**
 * GEE App for Water Mask Validation (Planet Scope)
 * Multi-site interactive viewer 
 */

// --- 1. GEOMETRIES ---
var aoiPozzillo = ee.Geometry.Polygon([[[14.554580696432122, 37.6766992581014], [14.554580696432122, 37.64897758060163], [14.617408760396966, 37.64897758060163], [14.617408760396966, 37.6766992581014]]], null, false);
var aoiAncipa = ee.Geometry.Polygon([[[14.57325691102672, 37.829510306031736], [14.57677597406553, 37.82944254240149], [14.578492583025685, 37.829239135754236], [14.579436720599281, 37.83144236541283], [14.57467311738862, 37.83520465165205], [14.563783306938234, 37.839356953333464], [14.562635380088457, 37.84113614580156], [14.56059688447167, 37.842321965613536], [14.558343834516467, 37.841203734129536], [14.559320198191177, 37.83911944092924], [14.556461064714243, 37.83885679609163], [14.552628213554826, 37.83909822359957], [14.550454245643296, 37.84033730950888], [14.548366011357844, 37.842457526834366], [14.544632370375181, 37.84549065349614], [14.543645323492353, 37.84588036505758], [14.541799931450177, 37.845049509355505], [14.540727189490793, 37.8433801988068], [14.544632672400146, 37.83746207809259], [14.552485884405758, 37.832560902982586], [14.562227667551046, 37.83398447172823], [14.56413743936066, 37.83385738061234], [14.56536054875059, 37.83318796014818], [14.567291678172841, 37.83042549836065], [14.568064220082166, 37.83015440239056], [14.570574736567504, 37.82905276680642]]]);
var aoiRosamarina = ee.Geometry.Polygon([[[13.608783486080712, 37.96112161679806], [13.607753517818994, 37.90669370285033], [13.654960396481103, 37.907100030549195], [13.654960396481103, 37.96112161679806]]], null, false);
var aoiPoma = aoiPoma2;
var aoiGarcia = ee.Geometry.Polygon([[[13.102364938493247, 37.78566159075255], [13.12777082228231, 37.79027407118122], [13.14751188063192, 37.80180401264191], [13.164334695573325, 37.81414583769675], [13.142362039323325, 37.81590878717127], [13.13051740431356, 37.8046523091972], [13.123307626481528, 37.804381047591185], [13.11129133009481, 37.79990508724729], [13.10356656813192, 37.8047879396266], [13.092751901383872, 37.80099019345061], [13.093953531022544, 37.79244455058606], [13.092408578629966, 37.786204250451654]]]);

// --- 2. ASSET DICTIONARY ---
var assetData = {
  'Pozzillo': {
    aoi: aoiPozzillo,
    images: {
      'Pozzillo - 2022/01/28': 'projects/ee-ciceromartinsjr/assets/planet/20220128_092614_pozzillo',
      'Pozzillo - 2024/05/28': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20240528_090325_13_24cf_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2024/06/10': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20240610_095627_82_24f4_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2024/06/23': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20240623_090406_01_24a8_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2024/07/17': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20240717_095520_74_24ee_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2024/07/28': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20240728_095448_62_24fd_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2024/08/15': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20240815_090922_76_24b3_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2024/09/06': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20240906_100129_30_24fa_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2024/10/08': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20241008_095732_82_24f4_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2024/11/02': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20241102_095946_56_24f4_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2024/11/14': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20241114_091837_54_24d0_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2024/12/19': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20241219_100534_82_24f7_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2024/12/31': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20241231_100352_85_24f9_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2025/01/10': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20250110_092502_68_24c4_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2025/01/21': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20250121_100219_74_24d9_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2025/02/12': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20250212_100311_03_2506_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2025/02/23': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20250223_092823_22_24bf_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2025/03/14': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20250314_100638_56_251b_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2025/04/18': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20250418_100600_58_2516_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2025/05/11': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20250511_101520_52_2538_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2025/05/18': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20250518_100614_64_251c_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2025/05/26': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20250526_101540_14_2527_3B_AnalyticMS_SR_clip',
      'Pozzillo - 2025/05/31': 'projects/ee-ciceromartinsjr/assets/planet/pozzillo/20250531_101718_22_2527_3B_AnalyticMS_SR_clip'
    }
  },
  'Ancipa': {
    aoi: aoiAncipa,
    images: {
      'Ancipa - 2022/01/28': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20220128_092612_ancipa',
      'Ancipa - 2022/05/22': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20220522_092258_ancipa',
      'Ancipa - 2023/12/26': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20231226_085749_ancipa',
      'Ancipa - 2024/04/20': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20240420_095420_65_24d7_3B_AnalyticMS_SR_clip',
      'Ancipa - 2024/04/29': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20240429_095851_69_24d5_3B_AnalyticMS_SR_clip',
      'Ancipa - 2024/05/05': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20240505_090339_27_24ab_3B_AnalyticMS_SR_clip',
      'Ancipa - 2024/05/24': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20240524_090624_95_24a1_3B_AnalyticMS_SR_clip',
      'Ancipa - 2024/12/19': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20241219_100532_ancipa',
      'Ancipa - 2024/12/31': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20241231_100350_64_24f9_3B_AnalyticMS_SR_clip',
      'Ancipa - 2025/01/30': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20250130_092618_99_24af_3B_AnalyticMS_SR_clip',
      'Ancipa - 2025/02/23': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20250223_092821_25_24bf_3B_AnalyticMS_SR_clip',
      'Ancipa - 2025/04/18': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20250418_100558_33_2516_3B_AnalyticMS_SR_clip',
      'Ancipa - 2025/05/03': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20250503_101738_80_252d_3B_AnalyticMS_SR_clip',
      'Ancipa - 2025/05/07': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20250507_100721_18_24f7_3B_AnalyticMS_SR_clip',
      'Ancipa - 2025/05/11': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20250511_101518_15_2538_3B_AnalyticMS_SR_clip',
      'Ancipa - 2025/05/17': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20250517_100552_92_250d_3B_AnalyticMS_SR_clip',
      'Ancipa - 2025/05/31': 'projects/ee-ciceromartinsjr/assets/planet/ancipa/20250531_101644_37_2522_3B_AnalyticMS_SR_clip'
    }
  },
  'Rosamarina': {
    aoi: aoiRosamarina,
    images: {
      'Rosamarina - 2022/01/09': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20220109_090019_rosamarina',
      'Rosamarina - 2023/09/02': 'projects/ee-ciceromartinsjr/assets/planet/20230902_085831_rosamarina',
      'Rosamarina - 2024/05/03': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240503_100208_74_24ed_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/05/17': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240517_100154_77_247a_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/05/23': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240523_100033_64_24f9_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/05/30': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240530_091013_82_24ba_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/06/06': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240606_100133_12_24cd_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/06/10': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240610_092001_42_2465_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/06/14': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240614_100720_70_2498_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/06/22': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240622_091621_08_2447_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/06/24': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240624_092703_70_242d_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/07/05': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240705_100215_11_24f3_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/07/16': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240716_100034_70_24cb_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/07/17': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240717_092843_82_2429_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/07/22': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240722_101354_66_2470_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/08/09': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240809_100254_65_24dd_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/08/10': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240810_091016_84_24c3_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/08/15': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240815_095927_29_24ed_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/09/26': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240926_091637_62_24bf_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/09/27': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20240927_092023_01_24ab_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/10/02': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20241002_091612_64_24bf_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/10/14': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20241014_091446_67_24a7_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/11/06': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20241106_100943_04_24db_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/11/13': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20241113_091619_65_24a7_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/11/14': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20241114_091833_82_24a8_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/11/18': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20241118_091846_73_24b6_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/12/12': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20241212_092154_26_2417_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/12/19': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20241219_091807_44_24a7_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2024/12/22': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20241222_092423_41_24c5_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2025/01/01': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20250101_100539_31_24dd_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2025/01/02': 'projects/ee-ciceromartinsjr/assets/planet/20250102_101018_rosamarina',
      'Rosamarina - 2025/01/05': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20250105_100814_25_24f6_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2025/01/18': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20250118_100732_60_2511_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2025/01/24': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20250124_100920_20_250e_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2025/01/25': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20250125_093336_87_24ab_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2025/03/14': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20250314_100743_41_24d7_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2025/04/23': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20250423_101215_89_250e_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2025/05/07': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20250507_100816_27_251a_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2025/05/18': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20250518_101217_78_2500_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2025/05/19': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20250519_101047_51_24fa_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2025/05/29': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20250529_101305_90_24ee_3B_AnalyticMS_SR_clip',
      'Rosamarina - 2025/05/30': 'projects/ee-ciceromartinsjr/assets/planet/rosamarina/20250530_101742_36_24f5_3B_AnalyticMS_SR_clip'
    }
  },
  'Poma': {
    aoi: aoiPoma,
    images: {
      'Poma - 2022/03/18': 'projects/ee-ciceromartinsjr/assets/planet/20220318_093238_poma',
      'Poma - 2023/09/03': 'projects/ee-ciceromartinsjr/assets/planet/20230903_090234_poma',
      'Poma - 2024/05/05': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240505_091326_92_24cf_3B_AnalyticMS_SR_clip',
      'Poma - 2024/05/17': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240517_090833_94_24ce_3B_AnalyticMS_SR_clip',
      'Poma - 2024/05/23': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240523_100435_78_24d7_3B_AnalyticMS_SR_clip',
      'Poma - 2024/05/29': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240529_091143_03_24c9_3B_AnalyticMS_SR_clip',
      'Poma - 2024/06/11': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240611_100646_62_248c_3B_AnalyticMS_SR_clip',
      'Poma - 2024/06/15': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240615_100036_26_24ee_3B_AnalyticMS_SR_clip',
      'Poma - 2024/06/29': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240629_091145_05_24a1_3B_AnalyticMS_SR_clip',
      'Poma - 2024/07/03': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240703_091133_79_24c3_3B_AnalyticMS_SR_clip',
      'Poma - 2024/07/10': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240710_100543_32_24f3_3B_AnalyticMS_SR_clip',
      'Poma - 2024/07/16': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240716_101224_29_247c_3B_AnalyticMS_SR_clip',
      'Poma - 2024/07/28': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240728_101423_03_2490_3B_AnalyticMS_SR_clip',
      'Poma - 2024/08/09': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240809_100505_76_24cb_3B_AnalyticMS_SR_clip',
      'Poma - 2024/08/14': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240814_092605_53_2420_3B_AnalyticMS_SR_clip',
      'Poma - 2024/08/27': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240827_100239_39_24f9_3B_AnalyticMS_SR_clip',
      'Poma - 2024/09/08': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240908_100425_78_24bd_3B_AnalyticMS_SR_clip',
      'Poma - 2024/09/20': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240920_092627_06_2415_3B_AnalyticMS_SR_clip',
      'Poma - 2024/09/26': 'projects/ee-ciceromartinsjr/assets/planet/poma/20240926_100332_05_24d1_3B_AnalyticMS_SR_clip',
      'Poma - 2024/10/13': 'projects/ee-ciceromartinsjr/assets/planet/poma/20241013_100910_19_250e_3B_AnalyticMS_SR_clip',
      'Poma - 2024/10/20': 'projects/ee-ciceromartinsjr/assets/planet/poma/20241020_100702_13_24f5_3B_AnalyticMS_SR_clip',
      'Poma - 2024/11/01': 'projects/ee-ciceromartinsjr/assets/planet/poma/20241101_100858_34_24dd_3B_AnalyticMS_SR_clip',
      'Poma - 2024/11/13': 'projects/ee-ciceromartinsjr/assets/planet/poma/20241113_100658_18_24dd_3B_AnalyticMS_SR_clip',
      'Poma - 2024/11/26': 'projects/ee-ciceromartinsjr/assets/planet/poma/20241126_101000_99_2507_3B_AnalyticMS_SR_clip',
      'Poma - 2024/12/07': 'projects/ee-ciceromartinsjr/assets/planet/poma/20241207_092427_98_24a8_3B_AnalyticMS_SR_clip',
      'Poma - 2024/12/27': 'projects/ee-ciceromartinsjr/assets/planet/20241227_092606_poma',
      'Poma - 2024/12/30': 'projects/ee-ciceromartinsjr/assets/planet/poma/20241230_100943_70_24f2_3B_AnalyticMS_SR_clip',
      'Poma - 2025/01/06': 'projects/ee-ciceromartinsjr/assets/planet/poma/20250106_101033_89_24f7_3B_AnalyticMS_SR_clip',
      'Poma - 2025/01/24': 'projects/ee-ciceromartinsjr/assets/planet/poma/20250124_100705_36_24ee_3B_AnalyticMS_SR_clip',
      'Poma - 2025/02/05': 'projects/ee-ciceromartinsjr/assets/planet/poma/20250205_093127_26_24b3_3B_AnalyticMS_SR_clip',
      'Poma - 2025/02/17': 'projects/ee-ciceromartinsjr/assets/planet/poma/20250217_100808_57_24da_3B_AnalyticMS_SR_clip',
      'Poma - 2025/02/24': 'projects/ee-ciceromartinsjr/assets/planet/poma/20250224_102206_10_251e_3B_AnalyticMS_SR_clip',
      'Poma - 2025/03/09': 'projects/ee-ciceromartinsjr/assets/planet/poma/20250309_093148_45_24a7_3B_AnalyticMS_SR_clip',
      'Poma - 2025/04/04': 'projects/ee-ciceromartinsjr/assets/planet/poma/20250404_101224_76_250b_3B_AnalyticMS_SR_clip',
      'Poma - 2025/04/15': 'projects/ee-ciceromartinsjr/assets/planet/poma/20250415_101640_61_24f2_3B_AnalyticMS_SR_clip',
      'Poma - 2025/04/19': 'projects/ee-ciceromartinsjr/assets/planet/poma/20250419_101717_51_24df_3B_AnalyticMS_SR_clip',
      'Poma - 2025/04/25': 'projects/ee-ciceromartinsjr/assets/planet/poma/20250425_101817_47_24aa_3B_AnalyticMS_SR_clip',
      'Poma - 2025/05/06': 'projects/ee-ciceromartinsjr/assets/planet/poma/20250506_101138_68_2409_3B_AnalyticMS_SR_clip',
      'Poma - 2025/05/10': 'projects/ee-ciceromartinsjr/assets/planet/poma/20250510_102326_06_2522_3B_AnalyticMS_SR_clip',
      'Poma - 2025/05/19': 'projects/ee-ciceromartinsjr/assets/planet/poma/20250519_101546_53_250e_3B_AnalyticMS_SR_clip',
      'Poma - 2025/05/29': 'projects/ee-ciceromartinsjr/assets/planet/poma/20250529_094829_79_24b9_3B_AnalyticMS_SR_clip'
    }
  }
  
};

// --- 2.1 PAPER THRESHOLDS TABLE ---
var paperThresholds = {
  // Ancipa
  'Ancipa - 2024/04/20': -0.1, 'Ancipa - 2024/04/29': -0.15, 'Ancipa - 2024/05/05': -0.15,
  'Ancipa - 2024/05/24': -0.15, 'Ancipa - 2024/06/23': -0.15, 'Ancipa - 2024/07/17': -0.15,
  'Ancipa - 2024/08/07': -0.1, 'Ancipa - 2024/08/27': -0.1, 'Ancipa - 2024/10/08': -0.1,
  'Ancipa - 2024/10/14': -0.1, 'Ancipa - 2024/10/28': -0.075, 'Ancipa - 2024/11/27': -0.075,
  'Ancipa - 2024/12/19': -0.075, 'Ancipa - 2024/12/31': -0.05, 'Ancipa - 2025/01/30': -0.05,
  'Ancipa - 2025/02/23': -0.2, 'Ancipa - 2025/03/14': -0.2, 'Ancipa - 2025/03/21': -0.2,
  'Ancipa - 2025/04/18': -0.25, 'Ancipa - 2025/04/29': -0.35, 'Ancipa - 2023/12/26': -0.25,
  'Ancipa - 2022/05/22': -0.4,
  
  // Poma
  'Poma - 2023/09/03': -0.15, 'Poma - 2024/05/05': 0, 'Poma - 2024/05/17': -0.15,
  'Poma - 2024/05/23': -0.1, 'Poma - 2024/05/29': -0.15, 'Poma - 2024/06/11': -0.1,
  'Poma - 2024/06/15': -0.1, 'Poma - 2024/06/29': -0.15, 'Poma - 2024/07/03': -0.1,
  'Poma - 2024/07/10': -0.1, 'Poma - 2024/07/16': -0.1, 'Poma - 2024/07/28': -0.1,
  'Poma - 2024/08/09': -0.1, 'Poma - 2024/08/14': -0.1, 'Poma - 2024/08/27': -0.1,
  'Poma - 2024/09/08': -0.1, 'Poma - 2024/09/20': -0.1, 'Poma - 2024/09/26': -0.1,
  'Poma - 2024/10/13': -0.1, 'Poma - 2024/10/20': -0.1, 'Poma - 2024/11/01': -0.1,
  'Poma - 2024/11/26': -0.1, 'Poma - 2024/12/07': -0.1, 'Poma - 2024/12/27': -0.1,
  'Poma - 2024/12/30': -0.1, 'Poma - 2025/01/06': -0.1, 'Poma - 2025/01/24': -0.1,
  'Poma - 2025/02/05': -0.1, 'Poma - 2025/02/17': -0.1, 'Poma - 2025/02/24': -0.1,
  'Poma - 2025/03/09': -0.1, 'Poma - 2025/04/04': -0.1, 'Poma - 2025/04/15': -0.25,
  'Poma - 2025/04/19': -0.1, 'Poma - 2025/04/25': -0.1, 'Poma - 2025/05/06': -0.2,
  'Poma - 2025/05/10': -0.2, 'Poma - 2025/05/19': -0.1, 'Poma - 2025/05/29': -0.1,

  // Pozzillo
  'Pozzillo - 2024/05/28': -0.1, 'Pozzillo - 2024/06/10': -0.1, 'Pozzillo - 2024/06/23': 0,
  'Pozzillo - 2024/07/17': -0.1, 'Pozzillo - 2024/07/28': -0.1, 'Pozzillo - 2024/08/15': -0.1,
  'Pozzillo - 2024/09/06': -0.1, 'Pozzillo - 2024/10/08': -0.05, 'Pozzillo - 2024/11/02': -0.1,
  'Pozzillo - 2024/11/14': -0.1, 'Pozzillo - 2024/12/19': -0.1, 'Pozzillo - 2024/12/31': -0.1,
  'Pozzillo - 2025/01/10': -0.2, 'Pozzillo - 2025/01/21': -0.2, 'Pozzillo - 2025/02/12': -0.1,
  'Pozzillo - 2025/02/23': -0.3, 'Pozzillo - 2025/03/14': -0.4, 'Pozzillo - 2025/04/18': -0.2,
  'Pozzillo - 2025/05/11': -0.2, 'Pozzillo - 2022/01/28': -0.25,

  // Rosamarina
  'Rosamarina - 2022/01/09': -0.1, 'Rosamarina - 2023/09/02': -0.1, 'Rosamarina - 2024/05/03': -0.1,
  'Rosamarina - 2024/05/17': -0.1, 'Rosamarina - 2024/05/23': -0.1, 'Rosamarina - 2024/05/30': -0.1,
  'Rosamarina - 2024/06/06': -0.1, 'Rosamarina - 2024/06/10': -0.1, 'Rosamarina - 2024/06/14': -0.1,
  'Rosamarina - 2024/06/22': -0.175, 'Rosamarina - 2024/06/24': -0.1, 'Rosamarina - 2024/07/05': -0.1,
  'Rosamarina - 2024/07/16': -0.1, 'Rosamarina - 2024/07/17': -0.1, 'Rosamarina - 2024/07/22': -0.1,
  'Rosamarina - 2024/08/09': -0.1, 'Rosamarina - 2024/08/10': -0.1, 'Rosamarina - 2024/08/15': -0.1,
  'Rosamarina - 2024/09/26': -0.05, 'Rosamarina - 2024/09/27': -0.1, 'Rosamarina - 2024/10/02': -0.1,
  'Rosamarina - 2024/10/14': -0.1, 'Rosamarina - 2024/11/06': -0.1, 'Rosamarina - 2024/11/13': -0.1,
  'Rosamarina - 2024/11/18': -0.1, 'Rosamarina - 2024/12/12': -0.1, 'Rosamarina - 2024/12/19': -0.1,
  'Rosamarina - 2024/12/22': -0.05, 'Rosamarina - 2025/01/01': -0.1, 'Rosamarina - 2025/01/05': -0.1,
  'Rosamarina - 2025/01/18': -0.1, 'Rosamarina - 2025/01/24': -0.1, 'Rosamarina - 2025/01/25': -0.1,
  'Rosamarina - 2025/03/14': -0.2, 'Rosamarina - 2025/04/23': -0.2, 'Rosamarina - 2025/05/07': -0.2,
  'Rosamarina - 2025/05/18': -0.1, 'Rosamarina - 2025/05/19': -0.1, 'Rosamarina - 2025/05/29': -0.1,
  'Rosamarina - 2025/05/30': -0.125, 'Rosamarina - 2025/01/02': -0.25,
};

// --- 3. UI COMPONENTS ---
var panel = ui.Panel({style: {width: '350px', padding: '10px'}});

var title = ui.Label('Water Mask Validation Tool', {fontWeight: 'bold', fontSize: '20px'});
var subTitle = ui.Label('Select a site and an image to visualize the classification results.');

// 3.1 Site Selector (with auto-centering)
var siteSelect = ui.Select({
  items: Object.keys(assetData),
  placeholder: '1. Select Area of Interest...',
  onChange: function(siteKey) {
    // A. Center map on the selected AOI immediately
    var aoi = assetData[siteKey].aoi;
    Map.centerObject(aoi, 14);
    
    // B. Clean current layers to avoid confusion
    Map.layers().reset();
    Map.addLayer(aoi, {color: 'red'}, 'AOI Boundary', false); // Add AOI as wireframe (hidden by default)

    // C. Update the second dropdown with corresponding images
    var siteImages = assetData[siteKey].images;
    imageSelect.items().reset(Object.keys(siteImages));
    imageSelect.setValue(null);
    
    // D. Reset result label
    resultLabel.setValue('Calculated Area: - ha');
  }
});

// 3.2 Image Selector
var imageSelect = ui.Select({
  placeholder: '2. Select Image Date...',
});

// 3.3 Threshold Slider
var thresholdSlider = ui.Slider({
  min: -1, max: 1, value: 0, step: 0.01,
  style: {stretch: 'horizontal'}
});

var runButton = ui.Button('Generate Results');
var resultLabel = ui.Label('Calculated Area: - ha', {fontWeight: 'bold', color: 'blue'});
var paperLabel = ui.Label('Paper Threshold: -', {fontWeight: 'normal', color: 'gray', fontSize: '12px'});

panel.add(title).add(subTitle)
     .add(ui.Label('Site Selection:', {fontWeight: 'bold'}))
     .add(siteSelect)
     .add(ui.Label('Available PlanetScope Images:', {fontWeight: 'bold'}))
     .add(imageSelect)
     .add(ui.Label('NDWI Threshold Adjustment:', {fontWeight: 'bold'}))
     .add(thresholdSlider)
     .add(runButton)
     .add(resultLabel)
     .add(paperLabel);

ui.root.insert(0, panel);

// --- 4. PROCESSING LOGIC ---
var process = function() {
  var siteKey = siteSelect.getValue();
  var dateKey = imageSelect.getValue();
  var threshold = thresholdSlider.getValue();
  
  if (!siteKey || !dateKey) {
    resultLabel.setValue('Error: Please select site AND image.');
    return;
  }
  
  var pThreshold = paperThresholds[dateKey];
  if (pThreshold !== undefined) {
    paperLabel.setValue('Threshold choosed in the paper: ' + pThreshold);
  } else {
    paperLabel.setValue('Paper Threshold: Not found for this date');
  }

  var aoi = assetData[siteKey].aoi;
  var assetPath = assetData[siteKey].images[dateKey];
  var imagePlanet = ee.Image(assetPath);

  // NDWI Calculation
  var ndwi = imagePlanet.normalizedDifference(['b2','b4']).rename('NDWI');
  
  // Water Masking
  var waterMask = ndwi.gt(threshold).clip(aoi).rename('water');
  
  // Area Calculation (3m res -> 9m² per pixel)
  var areaPixels = waterMask.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: aoi,
    scale: 3,
    maxPixels: 1e12
  }).get('water');

  var areaHa = ee.Number(areaPixels).multiply(9).divide(10000);
  
  // Update Map Display
  Map.layers().reset();
  Map.centerObject(aoi, 14);
  Map.addLayer(imagePlanet, {bands: ['b3', 'b2', 'b1'], min: 0, max: 2550}, 'Planet RGB');
  Map.addLayer(ndwi, {min: -0.5, max: 0.5, palette: ['red', 'white', 'blue']}, 'NDWI (Ref)', false);
  Map.addLayer(waterMask.selfMask(), {palette: ['0000FF']}, 'Final Water Mask');
  
  // Update UI Text
  areaHa.evaluate(function(val) {
    resultLabel.setValue('Calculated Area: ' + val.toFixed(2) + ' ha');
  });
};

runButton.onClick(process);
Map.style().set('cursor', 'crosshair');