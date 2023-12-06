-- init.sql

CREATE TABLE IF NOT EXISTS points (
	id 		SERIAL PRIMARY KEY,
 	poi_id		VARCHAR(32),
        tile_id		VARCHAR(5),	
	date 		TIMESTAMP,
	aws_index 	INTEGER,
	R20m_SCL  	INTEGER,
	R20m_B02  	INTEGER,
	R20m_B03  	INTEGER,
	R20m_B04  	INTEGER,       
	R20m_B05  	INTEGER,       
	R20m_B06  	INTEGER,       
	R20m_B07  	INTEGER,       
	R20m_B8A  	INTEGER,       
	R20m_B11  	INTEGER,       
	R20m_B12  	INTEGER       
);

CREATE TABLE IF NOT EXISTS labels (
	id 	SERIAL PRIMARY KEY,
 	poi_id 	VARCHAR(32),
	season 	INTEGER,
	label	VARCHAR(3) 
);

CREATE TABLE IF NOT EXISTS split (
	id 	SERIAL PRIMARY KEY,
 	poi_id 	VARCHAR(32),
	stage	VARCHAR(5)	
);

CREATE TABLE IF NOT EXISTS temperatures (
  	id 		SERIAL PRIMARY KEY,
 	poi_id 		VARCHAR(32),
	date 		TIMESTAMP,
        tile_id 	VARCHAR(5),	
  	temperature 	FLOAT
);

CREATE TABLE IF NOT EXISTS precipitations(
  	id 		SERIAL PRIMARY KEY,
 	poi_id 		VARCHAR(32),
	date 		TIMESTAMP,
        tile_id 	VARCHAR(5),	
  	volume		INTEGER
);
