DROP DATABASE instituto;
CREATE DATABASE instituto;
USE instituto;

CREATE TABLE alumnos (
	NIA INT NOT NULL,
    Fecha_de_nacimiento DATETIME,
    Telefono INT,
    PRIMARY KEY (NIA)
    );
    
CREATE TABLE asignatura (
	ID INT NOT NULL,
    Nombre VARCHAR(10),
    Profesor_DNI VARCHAR(20),
    PRIMARY KEY (ID)  
    );

CREATE TABLE cursa (
	alumnos_NIA INT,
    asignatura_ID INT,
    CONSTRAINT fk_alumn_curs
	FOREIGN KEY (alumnos_NIA) REFERENCES alumnos(NIA),
    CONSTRAINT fk_asign_curs
    FOREIGN KEY (asignatura_ID) REFERENCES asignatura(ID)
    );
    
CREATE TABLE profesor (
	DNI CHAR(9) NOT NULL,
    Nombre VARCHAR(20),
    Telefono INT, 
    Departamento INT,
    PRIMARY KEY (DNI)
    );

CREATE TABLE departamento (
	ID INT NOT NULL,
    Nombre VARCHAR(20),
    Jefe VARCHAR(20),
    PRIMARY KEY (ID)
    );
    

ALTER TABLE asignatura
ADD CONSTRAINT fk_profesor_asig
FOREIGN KEY (Profesor_DNI) REFERENCES profesor(DNI);

ALTER TABLE profesor
ADD CONSTRAINT fk_departamento
FOREIGN KEY (Departamento) REFERENCES departamento(ID);

ALTER TABLE departamento
ADD CONSTRAINT fk_profesor_dep
FOREIGN KEY (Jefe) REFERENCES profesor(DNI);

