## Database

没学过db。。。极速入门满足日常简单需求

[SQL Tutorial](https://www.w3schools.com/sql/default.asp)

```mysql
CREATE TABLE Persons (
    Personid int NOT NULL AUTO_INCREMENT,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255),
    Age int,
    PRIMARY KEY (Personid)
);
```

* NOT NULL
* AUTO INCREMENT