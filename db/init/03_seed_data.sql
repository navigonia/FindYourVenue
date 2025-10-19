-- Beispieltestdaten
INSERT INTO noise_points (geom, db_value, note)
VALUES (
  ST_SetSRID(ST_MakePoint(13.405, 52.52), 4326), 
  65, 
  'Testpunkt Berlin'
);

